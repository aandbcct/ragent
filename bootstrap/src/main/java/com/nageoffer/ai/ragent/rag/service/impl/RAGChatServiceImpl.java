/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.rag.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.framework.context.UserContext;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceContext;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.infra.chat.StreamCallback;
import com.nageoffer.ai.ragent.infra.chat.StreamCancellationHandle;
import com.nageoffer.ai.ragent.rag.aop.ChatRateLimit;
import com.nageoffer.ai.ragent.rag.core.guidance.GuidanceDecision;
import com.nageoffer.ai.ragent.rag.core.guidance.IntentGuidanceService;
import com.nageoffer.ai.ragent.rag.core.intent.IntentResolver;
import com.nageoffer.ai.ragent.rag.core.memory.ConversationMemoryService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptContext;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import com.nageoffer.ai.ragent.rag.core.prompt.RAGPromptService;
import com.nageoffer.ai.ragent.rag.core.retrieve.RetrievalEngine;
import com.nageoffer.ai.ragent.rag.core.rewrite.QueryRewriteService;
import com.nageoffer.ai.ragent.rag.core.rewrite.RewriteResult;
import com.nageoffer.ai.ragent.rag.dto.IntentGroup;
import com.nageoffer.ai.ragent.rag.dto.RetrievalContext;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.rag.service.RAGChatService;
import com.nageoffer.ai.ragent.rag.service.handler.StreamCallbackFactory;
import com.nageoffer.ai.ragent.rag.service.handler.StreamTaskManager;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.ArrayList;
import java.util.List;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.CHAT_SYSTEM_PROMPT_PATH;
import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.DEFAULT_TOP_K;

/**
 * RAG 对话服务默认实现
 * <p>
 * 核心流程：
 * 记忆加载 -> 改写拆分 -> 意图解析 -> 歧义引导 -> 检索(MCP+KB) -> Prompt 组装 -> 流式输出
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RAGChatServiceImpl implements RAGChatService {

    private final LLMService llmService; // 负责调用大模型并流式返回结果
    private final RAGPromptService promptBuilder; // 负责拼装结构化 Prompt 消息
    private final PromptTemplateLoader promptTemplateLoader; // 负责加载系统提示词模板
    private final ConversationMemoryService memoryService; // 负责多轮会话记忆读写
    private final StreamTaskManager taskManager;
    private final IntentGuidanceService guidanceService; // 负责歧义检测与澄清引导
    private final StreamCallbackFactory callbackFactory; // 负责创建 SSE 回调处理器
    private final QueryRewriteService queryRewriteService; // 负责问题改写与多问拆分
    private final IntentResolver intentResolver; // 负责子问题意图识别与聚合
    private final RetrievalEngine retrievalEngine; // 负责 MCP/知识库检索

    @Override
    @ChatRateLimit
    public void streamChat(String question, String conversationId, Boolean deepThinking, SseEmitter emitter) {
        // 1) 统一会话 ID 与任务 ID：外部未传时自动生成，便于链路追踪和中断任务
        String actualConversationId = StrUtil.isBlank(conversationId) ? IdUtil.getSnowflakeNextIdStr() : conversationId;
        String taskId = StrUtil.isBlank(RagTraceContext.getTaskId())
                ? IdUtil.getSnowflakeNextIdStr()
                : RagTraceContext.getTaskId();
        log.info("开始流式对话，会话ID：{}，任务ID：{}", actualConversationId, taskId);
        boolean thinkingEnabled = Boolean.TRUE.equals(deepThinking);

        // 2) 创建流式回调：统一向前端推送 token、完成和异常事件
        StreamCallback callback = callbackFactory.createChatEventHandler(emitter, actualConversationId, taskId);

        // 3) 读取并追加会话记忆：将当前问题作为最新 user 消息写入上下文
        String userId = UserContext.getUserId();
        List<ChatMessage> history = memoryService.loadAndAppend(actualConversationId, userId, ChatMessage.user(question));

        // 4) 改写 + 拆分问题，并做子问题意图识别，为后续检索与 Prompt 规划提供输入
        RewriteResult rewriteResult = queryRewriteService.rewriteWithSplit(question, history);
        List<SubQuestionIntent> subIntents = intentResolver.resolve(rewriteResult);

        // 5) 歧义引导分支：若存在冲突意图，优先追问澄清后再继续
        GuidanceDecision guidanceDecision = guidanceService.detectAmbiguity(rewriteResult.rewrittenQuestion(), subIntents);
        if (guidanceDecision.isPrompt()) {
            callback.onContent(guidanceDecision.getPrompt());
            callback.onComplete();
            return;
        }

        // 6) system-only 快路径：若全部为系统能力，跳过检索直接回答
        boolean allSystemOnly = subIntents.stream()
                .allMatch(si -> intentResolver.isSystemOnly(si.nodeScores()));
        if (allSystemOnly) {
            // 优先使用意图节点自带模板，缺失时在下游方法回退到默认系统模板
            String customPrompt = subIntents.stream()
                    .flatMap(si -> si.nodeScores().stream())
                    .map(ns -> ns.getNode().getPromptTemplate())
                    .filter(StrUtil::isNotBlank)
                    .findFirst()
                    .orElse(null);
            StreamCancellationHandle handle = streamSystemResponse(rewriteResult.rewrittenQuestion(), history, customPrompt, callback);
            // 绑定任务句柄，支持 stopTask 主动中断
            taskManager.bindHandle(taskId, handle);
            return;
        }

        // 7) 常规 RAG 路径：检索 MCP 与知识库上下文
        RetrievalContext ctx = retrievalEngine.retrieve(subIntents, DEFAULT_TOP_K);
        // 8) 检索为空兜底：直接反馈未命中文档，避免无依据生成
        if (ctx.isEmpty()) {
            String emptyReply = "未检索到与问题相关的文档内容。";
            callback.onContent(emptyReply);
            callback.onComplete();
            return;
        }

        // 9) 聚合多子问题意图，形成统一的 Prompt 规划输入
        IntentGroup mergedGroup = intentResolver.mergeIntentGroup(subIntents);

        // 10) 构造请求并发起流式生成，最后绑定取消句柄
        StreamCancellationHandle handle = streamLLMResponse(
                rewriteResult,
                ctx,
                mergedGroup,
                history,
                thinkingEnabled,
                callback
        );
        taskManager.bindHandle(taskId, handle);
    }

    @Override
    public void stopTask(String taskId) {
        // 通过任务管理器中断正在进行的流式任务
        taskManager.cancel(taskId);
    }

    // ==================== LLM 响应 ====================

    private StreamCancellationHandle streamSystemResponse(String question, List<ChatMessage> history,
                                                          String customPrompt, StreamCallback callback) {
        // system-only 场景：优先使用自定义模板，否则加载默认系统模板
        String systemPrompt = StrUtil.isNotBlank(customPrompt)
                ? customPrompt
                : promptTemplateLoader.load(CHAT_SYSTEM_PROMPT_PATH);

        // 消息顺序：system + 历史(不含当前问题) + 当前 user 问题
        List<ChatMessage> messages = new ArrayList<>();
        messages.add(ChatMessage.system(systemPrompt));
        if (CollUtil.isNotEmpty(history)) {
            messages.addAll(history.subList(0, history.size() - 1));
        }
        messages.add(ChatMessage.user(question));

        // system-only 默认关闭深度思考，温度适中保证表达自然
        ChatRequest req = ChatRequest.builder()
                .messages(messages)
                .temperature(0.7D)
                .thinking(false)
                .build();
        return llmService.streamChat(req, callback);
    }

    private StreamCancellationHandle streamLLMResponse(RewriteResult rewriteResult, RetrievalContext ctx,
                                                       IntentGroup intentGroup, List<ChatMessage> history,
                                                       boolean deepThinking, StreamCallback callback) {
        // 整合改写问题、检索片段和意图分组，形成 Prompt 上下文
        PromptContext promptContext = PromptContext.builder()
                .question(rewriteResult.rewrittenQuestion())
                .mcpContext(ctx.getMcpContext())
                .kbContext(ctx.getKbContext())
                .mcpIntents(intentGroup.mcpIntents())
                .kbIntents(intentGroup.kbIntents())
                .intentChunks(ctx.getIntentChunks())
                .build();

        // 由 Prompt 服务生成结构化消息（主问题 + 子问题 + 历史）
        List<ChatMessage> messages = promptBuilder.buildStructuredMessages(
                promptContext,
                history,
                rewriteResult.rewrittenQuestion(),
                rewriteResult.subQuestions()  // 传入子问题列表
        );
        // MCP 场景适当放宽采样参数；纯 KB 场景更偏确定性输出
        ChatRequest chatRequest = ChatRequest.builder()
                .messages(messages)
                .thinking(deepThinking)
                .temperature(ctx.hasMcp() ? 0.3D : 0D)  // MCP 场景稍微放宽温度
                .topP(ctx.hasMcp() ? 0.8D : 1D)
                .build();

        return llmService.streamChat(chatRequest, callback);
    }
}
