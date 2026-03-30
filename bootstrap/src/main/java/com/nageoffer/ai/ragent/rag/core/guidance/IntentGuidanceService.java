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

package com.nageoffer.ai.ragent.rag.core.guidance;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.StrUtil;
import com.nageoffer.ai.ragent.rag.config.GuidanceProperties;
import com.nageoffer.ai.ragent.rag.constant.RAGConstant;
import com.nageoffer.ai.ragent.rag.dto.SubQuestionIntent;
import com.nageoffer.ai.ragent.rag.enums.IntentLevel;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNode;
import com.nageoffer.ai.ragent.rag.core.intent.IntentNodeRegistry;
import com.nageoffer.ai.ragent.rag.core.intent.NodeScore;
import com.nageoffer.ai.ragent.rag.core.prompt.PromptTemplateLoader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class IntentGuidanceService {

    private final GuidanceProperties guidanceProperties;
    private final IntentNodeRegistry intentNodeRegistry;
    private final PromptTemplateLoader promptTemplateLoader;

    public GuidanceDecision detectAmbiguity(String question, List<SubQuestionIntent> subIntents) {
        // 1) 开关关闭：直接跳过引导，保持原流程回答
        if (!Boolean.TRUE.equals(guidanceProperties.getEnabled())) {
            return GuidanceDecision.none();
        }

        // 2) 从意图结果中提取“可引导歧义组”（同主题、多系统、分数接近）
        AmbiguityGroup group = findAmbiguityGroup(subIntents);
        if (group == null || CollUtil.isEmpty(group.optionIds())) {
            return GuidanceDecision.none();
        }

        // 3) 若用户问题里已明确提到系统别名，则不再追问，避免重复打断
        List<String> systemNames = resolveOptionNames(group.optionIds());
        if (shouldSkipGuidance(question, systemNames)) {
            return GuidanceDecision.none();
        }

        // 4) 生成澄清提示词，返回“需要追问”决策
        String prompt = buildPrompt(group.topicName(), group.optionIds());
        return GuidanceDecision.prompt(prompt);
    }

    private AmbiguityGroup findAmbiguityGroup(List<SubQuestionIntent> subIntents) {
        // 当前仅对“单子问题”场景做歧义引导，避免多问句场景误触发
        if (CollUtil.isEmpty(subIntents) || subIntents.size() != 1) {
            return null;
        }

        // 过滤掉低分与非 KB 节点，保留可比较的候选意图
        List<NodeScore> candidates = filterCandidates(subIntents.get(0).nodeScores());
        if (candidates.size() < 2) {
            return null;
        }

        // 按归一化名称分组，识别“同主题下的多个候选节点”
        Map<String, List<NodeScore>> grouped = candidates.stream()
                .filter(ns -> StrUtil.isNotBlank(ns.getNode().getName()))
                .collect(Collectors.groupingBy(ns -> normalizeName(ns.getNode().getName())));

        // 选择最可能的歧义组：同组至少2个、分数比达阈值、且落在不同系统
        Optional<Map.Entry<String, List<NodeScore>>> best = grouped.entrySet().stream()
                .map(entry -> Map.entry(entry.getKey(), sortByScore(entry.getValue())))
                .filter(entry -> entry.getValue().size() > 1)
                .filter(entry -> passScoreRatio(entry.getValue()))
                .filter(entry -> hasMultipleSystems(entry.getValue()))
                .max(Comparator.comparingDouble(entry -> entry.getValue().get(0).getScore()));

        if (best.isEmpty()) {
            return null;
        }

        List<NodeScore> groupScores = best.get().getValue();
        String topicName = Optional.ofNullable(groupScores.get(0).getNode().getName())
                .orElse(best.get().getKey());
        // 抽取可供用户选择的系统级选项（去重、保序）
        List<String> optionIds = collectSystemOptions(groupScores);
        if (optionIds.size() < 2) {
            return null;
        }
        // 控制最大选项数量，避免追问过长
        return new AmbiguityGroup(topicName, trimOptions(optionIds));
    }

    private List<NodeScore> filterCandidates(List<NodeScore> scores) {
        if (CollUtil.isEmpty(scores)) {
            return List.of();
        }
        // 仅保留达到最小分数阈值且属于知识库链路的候选
        return scores.stream()
                .filter(ns -> ns.getScore() >= RAGConstant.INTENT_MIN_SCORE)
                .filter(ns -> ns.getNode() != null && ns.getNode().isKB())
                .toList();
    }

    private List<String> collectSystemOptions(List<NodeScore> groupScores) {
        // 使用 LinkedHashSet：保证去重同时保留“按得分排序后的插入顺序”
        Set<String> ordered = new LinkedHashSet<>();
        for (NodeScore score : groupScores) {
            IntentNode node = score.getNode();
            // 把任意层级节点回溯到“系统/分类”层，作为最终可选项
            String systemId = resolveSystemNodeId(node);
            if (StrUtil.isNotBlank(systemId)) {
                ordered.add(systemId);
            }
        }
        return new ArrayList<>(ordered);
    }

    private boolean shouldSkipGuidance(String question, List<String> systemNames) {
        if (StrUtil.isBlank(question) || CollUtil.isEmpty(systemNames)) {
            return false;
        }
        // 对问题和系统名做统一归一化，再做包含匹配
        String normalizedQuestion = normalizeName(question);
        for (String name : systemNames) {
            if (StrUtil.isBlank(name)) {
                continue;
            }
            for (String alias : buildSystemAliases(name)) {
                if (alias.length() < 2) {
                    continue;
                }
                if (normalizedQuestion.contains(alias)) {
                    return true;
                }
            }
        }
        return false;
    }

    private List<String> resolveOptionNames(List<String> optionIds) {
        if (CollUtil.isEmpty(optionIds)) {
            return List.of();
        }
        List<String> names = new ArrayList<>();
        for (String id : optionIds) {
            // 使用注册表把系统 ID 转为展示名称，不存在时跳过
            IntentNode node = intentNodeRegistry.getNodeById(id);
            if (node == null) {
                continue;
            }
            String name = StrUtil.blankToDefault(node.getName(), node.getId());
            names.add(name);
        }
        return names;
    }

    private List<String> buildSystemAliases(String systemName) {
        if (StrUtil.isBlank(systemName)) {
            return List.of();
        }
        // 目前别名策略为“归一化系统名本身”，后续可扩展同义词
        String normalized = normalizeName(systemName);
        List<String> aliases = new ArrayList<>();
        if (StrUtil.isNotBlank(normalized)) {
            aliases.add(normalized);
        }
        return aliases;
    }

    private boolean passScoreRatio(List<NodeScore> group) {
        if (group.size() < 2) {
            return false;
        }
        // 用第二名/第一名分数比衡量“歧义程度”，越接近越需要引导
        double top = group.get(0).getScore();
        double second = group.get(1).getScore();
        if (top <= 0) {
            return false;
        }
        double ratio = second / top;
        return ratio >= Optional.ofNullable(guidanceProperties.getAmbiguityScoreRatio()).orElse(0.0D);
    }

    private boolean hasMultipleSystems(List<NodeScore> group) {
        // 必须跨系统冲突才触发引导，同系统内竞争不追问
        Set<String> systems = group.stream()
                .map(NodeScore::getNode)
                .map(this::resolveSystemNodeId)
                .filter(StrUtil::isNotBlank)
                .collect(Collectors.toSet());
        return systems.size() > 1;
    }

    private String resolveSystemNodeId(IntentNode node) {
        if (node == null) {
            return "";
        }
        // 自底向上回溯：优先定位到 CATEGORY（其父为空或为 DOMAIN）作为系统标识
        IntentNode current = node;
        IntentNode parent = fetchParent(current);
        for (; ; ) {
            IntentLevel level = current.getLevel();
            if (level == IntentLevel.CATEGORY && (parent == null || parent.getLevel() == IntentLevel.DOMAIN)) {
                return current.getId();
            }
            // 回溯到根仍未命中 CATEGORY 时，退化使用当前节点 ID
            if (parent == null) {
                return current.getId();
            }
            current = parent;
            parent = fetchParent(current);
        }
    }

    private IntentNode fetchParent(IntentNode node) {
        if (node == null || StrUtil.isBlank(node.getParentId())) {
            return null;
        }
        // 通过节点注册表按 parentId 查询父节点
        return intentNodeRegistry.getNodeById(node.getParentId());
    }

    private List<NodeScore> sortByScore(List<NodeScore> scores) {
        // 按得分降序，确保 top1/top2 可直接用于分数比判断
        return scores.stream()
                .sorted(Comparator.comparingDouble(NodeScore::getScore).reversed())
                .toList();
    }

    private List<String> trimOptions(List<String> optionIds) {
        // 最多返回配置限定数量的候选项，降低用户选择负担
        int maxOptions = Optional.ofNullable(guidanceProperties.getMaxOptions()).orElse(optionIds.size());
        if (optionIds.size() <= maxOptions) {
            return optionIds;
        }
        return optionIds.subList(0, maxOptions);
    }

    private String buildPrompt(String topicName, List<String> optionIds) {
        // 渲染引导模板：注入主题名与候选系统列表
        String options = renderOptions(optionIds);
        return promptTemplateLoader.render(
                RAGConstant.GUIDANCE_PROMPT_PATH,
                Map.of(
                        "topic_name", StrUtil.blankToDefault(topicName, ""),
                        "options", options
                )
        );
    }

    private String renderOptions(List<String> optionIds) {
        // 将候选系统渲染为“1) xxx”形式，供追问提示词直接使用
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < optionIds.size(); i++) {
            String id = optionIds.get(i);
            IntentNode node = intentNodeRegistry.getNodeById(id);
            String name = node == null || StrUtil.isBlank(node.getName()) ? id : node.getName();
            sb.append(i + 1).append(") ").append(name).append("\n");
        }
        return sb.toString().trim();
    }

    private String normalizeName(String name) {
        if (name == null) {
            return "";
        }
        // 统一小写并移除空白/标点，提升字符串匹配鲁棒性
        String cleaned = name.trim().toLowerCase(Locale.ROOT);
        return cleaned.replaceAll("[\\p{Punct}\\s]+", "");
    }

    private record AmbiguityGroup(String topicName, List<String> optionIds) {
    }
}
