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

package com.nageoffer.ai.ragent.rag.core.rewrite;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.nageoffer.ai.ragent.infra.util.LLMResponseCleaner;
import com.nageoffer.ai.ragent.rag.config.RAGConfigProperties;
import com.nageoffer.ai.ragent.framework.convention.ChatMessage;
import com.nageoffer.ai.ragent.framework.convention.ChatRequest;
import com.nageoffer.ai.ragent.framework.trace.RagTraceNode;
import com.nageoffer.ai.ragent.infra.chat.LLMService;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.nageoffer.ai.ragent.rag.constant.RAGConstant.QUERY_REWRITE_AND_SPLIT_PROMPT_PATH;

/**
 * 查询预处理：改写 + 拆分多问句
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MultiQuestionRewriteService implements QueryRewriteService {

    private final LLMService llmService; // 负责调用大模型完成改写与拆分
    private final RAGConfigProperties ragConfigProperties; // 查询改写相关开关配置
    private final QueryTermMappingService queryTermMappingService; // 术语归一化服务
    private final PromptTemplateLoader promptTemplateLoader; // 改写提示词模板加载器

    @Override
    @RagTraceNode(name = "query-rewrite", type = "REWRITE")
    public String rewrite(String userQuestion) {
        // 仅返回改写后的主问题（兼容只需要 rewrite 字段的调用方）
        return rewriteAndSplit(userQuestion).rewrittenQuestion();
    }

    @Override
    public RewriteResult rewriteWithSplit(String userQuestion) {
        // 无历史上下文场景：走统一的改写+拆分流程
        return rewriteAndSplit(userQuestion);
    }

    @Override
    @RagTraceNode(name = "query-rewrite-and-split", type = "REWRITE")
    public RewriteResult rewriteWithSplit(String userQuestion, List<ChatMessage> history) {
        // 1) 关闭改写开关时：仅做术语归一化 + 规则拆分，避免额外 LLM 开销
        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            String normalized = queryTermMappingService.normalize(userQuestion);
            List<String> subs = ruleBasedSplit(normalized);
            return new RewriteResult(normalized, subs);
        }

        // 2) 先做术语归一化，降低 LLM 改写歧义并提升可检索性
        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        // 3) 携带历史上下文调用 LLM 执行“改写 + 子问题拆分”
        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, history);
    }

    /**
     * 先用默认改写做归一化，再进行多问句拆分。
     */
    private RewriteResult rewriteAndSplit(String userQuestion) {
        // 开关关闭：直接做规则归一化 + 规则拆分
        if (!ragConfigProperties.getQueryRewriteEnabled()) {
            String normalized = queryTermMappingService.normalize(userQuestion);
            List<String> subs = ruleBasedSplit(normalized);
            return new RewriteResult(normalized, subs);
        }

        // 开关开启：先归一化，再由 LLM 输出结构化 rewrite/sub_questions
        String normalizedQuestion = queryTermMappingService.normalize(userQuestion);

        // 无历史场景传空历史，保持与带历史版本一致的处理逻辑
        return callLLMRewriteAndSplit(normalizedQuestion, userQuestion, List.of());

        // 兜底：使用归一化结果 + 规则拆分
    }

    private RewriteResult callLLMRewriteAndSplit(String normalizedQuestion,
                                                 String originalQuestion,
                                                 List<ChatMessage> history) {
        // 1) 加载系统提示词并构造改写请求
        String systemPrompt = promptTemplateLoader.load(QUERY_REWRITE_AND_SPLIT_PROMPT_PATH);
        ChatRequest req = buildRewriteRequest(systemPrompt, normalizedQuestion, history);

        try {
            // 2) 调用 LLM 并解析结构化结果
            String raw = llmService.chat(req);
            RewriteResult parsed = parseRewriteAndSplit(raw);

            // 3) 解析成功则返回，并输出关键链路日志
            if (parsed != null) {
                log.info("""
                        RAG用户问题查询改写+拆分：
                        原始问题：{}
                        归一化后：{}
                        改写结果：{}
                        子问题：{}
                        """, originalQuestion, normalizedQuestion, parsed.rewrittenQuestion(), parsed.subQuestions());
                return parsed;
            }

            // 4) 解析失败：走统一兜底
            log.warn("查询改写+拆分解析失败，使用归一化问题兜底 - normalizedQuestion={}", normalizedQuestion);
        } catch (Exception e) {
            // 5) 调用异常：走统一兜底，保障主流程稳定
            log.warn("查询改写+拆分 LLM 调用失败，使用归一化问题兜底 - question={}，normalizedQuestion={}", originalQuestion, normalizedQuestion, e);
        }

        // 统一兜底逻辑
        return new RewriteResult(normalizedQuestion, List.of(normalizedQuestion));
    }

    private ChatRequest buildRewriteRequest(String systemPrompt,
                                            String question,
                                            List<ChatMessage> history) {
        // 消息结构：system + 最近历史 + 当前问题
        List<ChatMessage> messages = new ArrayList<>();
        if (StrUtil.isNotBlank(systemPrompt)) {
            messages.add(ChatMessage.system(systemPrompt));
        }

        // 只保留最近 1-2 轮的 User 和 Assistant 消息
        // 过滤掉 System 摘要，避免 Token 浪费
        if (CollUtil.isNotEmpty(history)) {
            List<ChatMessage> recentHistory = history.stream()
                    // 仅保留 user/assistant，过滤 system 摘要以节省 token
                    .filter(msg -> msg.getRole() == ChatMessage.Role.USER
                            || msg.getRole() == ChatMessage.Role.ASSISTANT)
                    .skip(Math.max(0, history.size() - 4))  // 最多保留最近 4 条消息（2 轮对话）
                    .toList();
            messages.addAll(recentHistory);
        }

        messages.add(ChatMessage.user(question));

        // 改写任务采用低温参数，优先稳定与可复现
        return ChatRequest.builder()
                .messages(messages)
                .temperature(0.1D)
                .topP(0.3D)
                .thinking(false)
                .build();
    }


    private RewriteResult parseRewriteAndSplit(String raw) {
        try {
            // 移除可能存在的 Markdown 代码块标记
            String cleaned = LLMResponseCleaner.stripMarkdownCodeFence(raw);

            // 1) 解析 JSON 根节点，非对象直接视为失败
            JsonElement root = JsonParser.parseString(cleaned);
            if (!root.isJsonObject()) {
                return null;
            }
            // 2) 提取 rewrite 与 sub_questions 字段
            JsonObject obj = root.getAsJsonObject();
            String rewrite = obj.has("rewrite") ? obj.get("rewrite").getAsString().trim() : "";
            List<String> subs = new ArrayList<>();
            if (obj.has("sub_questions") && obj.get("sub_questions").isJsonArray()) {
                JsonArray arr = obj.getAsJsonArray("sub_questions");
                for (JsonElement el : arr) {
                    if (el.isJsonPrimitive() && el.getAsJsonPrimitive().isString()) {
                        String s = el.getAsString().trim();
                        if (StrUtil.isNotBlank(s)) {
                            subs.add(s);
                        }
                    }
                }
            }
            // 3) rewrite 为空则判定无效
            if (StrUtil.isBlank(rewrite)) {
                return null;
            }
            // 4) 若未提供子问题，默认退化为“单子问题=改写问题”
            if (CollUtil.isEmpty(subs)) {
                subs = List.of(rewrite);
            }
            return new RewriteResult(rewrite, subs);
        } catch (Exception e) {
            // 5) 解析异常返回 null，由上层统一兜底
            log.warn("解析改写+拆分结果失败，raw={}", raw, e);
            return null;
        }
    }

    private List<String> ruleBasedSplit(String question) {
        // 兜底：按常见分隔符拆分
        List<String> parts = Arrays.stream(question.split("[?？。；;\\n]+"))
                .map(String::trim)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toList());

        // 拆分后为空时保留原问题，避免下游出现空列表
        if (CollUtil.isEmpty(parts)) {
            return List.of(question);
        }
        // 统一补齐问号，便于下游按“问题句”处理
        return parts.stream()
                .map(s -> s.endsWith("？") || s.endsWith("?") ? s : s + "？")
                .toList();
    }
}
