# -*- coding: utf-8 -*-
import asyncio
import json
from typing import List, Dict, Tuple, Optional
import ast
import re
from asyncio import to_thread

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult

# ==============================================================================
# --- Prompt Definitions (Enhanced for Clarity and Uniqueness) ---
# ==============================================================================

PLANNING_PROMPT = """
Generate a detailed, structured outline (Table of Contents) for a formal, thesis-level research report addressing the main query and incorporating the sub-queries. Use the history context for insight if relevant. Ensure each section is unique and does not overlap in content with other sections. Avoid any repetition or duplication of topics across sections.

The report should follow a standard academic structure:
1. Introduction (Background, Problem Statement, Objectives, Scope, Report Structure)
2. Literature Review (Optional, if relevant)
3. Methodology (If applicable)
4. Analysis / Discussion Sections (Logically organized, addressing sub-queries)
5. Conclusion (Summary, implications, limitations, future work)
6. References (List of sources, and full URLs if available)

Query: {query}
Sub-Queries: {sub_queries}
History Context (last 30k chars): {history_context}

Respond *only* with a list of sections in the format:
- Section Title: Brief description of content and relevant sub-queries covered.
"""

DRAFT_SECTION_PROMPT = """
Draft a comprehensive, formal, and analytical section for the research report based on the provided outline section, relevant sub-queries, and retrieved information. Use a formal academic tone, avoid redundancy with other sections, and keep the content focused.

**Instructions:**
- **Unique Focus**: Ensure the content uniquely addresses the specific focus of '{section}' as described in the outline: '{plan}'. Do not repeat information that belongs in other sections.
- **Deep Analysis**: Critically analyze the information in detail and provide unique and professional insights. You may conduct hypothetical analysis but must highlight it. Do NOT provide analysis results that cannot be conducted (e.g., Monte Carlo simulations) unless supported by retrieved information or logical reasoning.
- **Critical Thinking**: If counterintuitive or contradictory information is detected, evaluate it carefully and proceed with scientific reasoning.
- **Synthesis**: Blend multiple sources into a cohesive narrative.
- **Source Attribution**: Cite sources as <N>.
- **Context**: Align with the report structure: {plan_summary}.
- **Original Query**: Address: {original_query}.

Section Title: {section}
Section Description: {plan}
Relevant Sub-Queries: {relevant_sub_queries}
Retrieved Information:
{retrieved_info}

Respond *only* with the drafted section in Markdown format (e.g., '## {section}'). If and only if the section title is 'References', list each source, with its full URL if available.
"""

POLISH_TRANSITION_PROMPT = """
Evaluate the transition between the end of the first section and the beginning of the second section. If the transition is not smooth, generate a bridge paragraph to be appended to the end of the first section to improve the flow.

**Instructions:**
- **Evaluation**: Assess whether the last {n_sentences} sentences of the first section and the first {n_sentences} sentences of the second section form a logical and smooth transition.
- **Content Addition**: If the transition is not smooth, create a bridge paragraph to be added at the end of the first section. This should enhance the connection to the second section without modifying existing content. Ensure the bridge paragraph does not repeat content from either section but instead highlights the logical connection between them.
- **No Change Needed**: If the transition is already smooth, respond with an empty string.
- **Output**: Return only the bridge paragraph (in Markdown format) or an empty string. NEVER create references or citations in the bridge paragraph.

First Section (Last {n_sentences} Sentences):
{first_section_end}

Second Section (First {n_sentences} Sentences):
{second_section_start}
"""

MAPPING_PROMPT = """
Select the most relevant sub-queries for the section based on its title and description. Aim to assign each sub-query to sections where relevant.

Section Title: {section}
Section Description: {description}
Available Sub-Queries: {sub_queries}

Respond *only* with relevant sub-queries, each on a new line starting with '- '. If none apply, respond with "None".
"""

FALLBACK_MAPPING_PROMPT = """
Rank sub-queries by relevance to the section and return the top 2.

Section: {section}
Description: {description}
Sub-Queries: {sub_queries}

Respond *only* with the top 2 sub-queries, each on a new line starting with '- '. If none apply, respond with "None".
"""

# ==============================================================================
# --- Agent Class Definition ---
# ==============================================================================

@describe_class(
    "FinalPaperAgent generates thesis-level structured reports with planning, drafting, and refined sentence-based transitions."
)
class FinalPaperAgent(RAGAgent):
    DEFAULT_TIMEOUTS = {
        "plan": 180, "map": 180, "draft": 600, "polish": 600,
    }

    def __init__(self,
                 lightllm: BaseLLM,
                 highllm: BaseLLM,
                 max_concurrent_tasks: int = 10,
                 transition_sentences: int = 3,
                 timeouts: Optional[Dict[str, int]] = None):
        self.lightllm = lightllm
        self.highllm = highllm
        self.max_concurrent_tasks = max(1, max_concurrent_tasks)  # Ensure at least 1
        self.transition_sentences = max(1, transition_sentences)  # Ensure at least 1
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.timeouts = self.DEFAULT_TIMEOUTS.copy()
        if timeouts:
            self.timeouts.update(timeouts)

        log.info(f"Initialized with timeouts: {self.timeouts}, concurrency: {self.max_concurrent_tasks}, transition sentences: {self.transition_sentences}")

    async def _call_llm(self, llm: BaseLLM, prompt: str, task_name: str, role: str = "user") -> str:
        base_task_name = task_name.split('_')[0]
        timeout = self.timeouts.get(base_task_name, 60)

        async with self.semaphore:
            try:
                log.debug(f"Starting LLM call for '{task_name}' with timeout {timeout}s.")
                response = await asyncio.wait_for(llm.chat_async([{"role": role, "content": prompt}]), timeout=timeout)
                content = getattr(response, 'content', '').strip()
                if not content:
                    log.warning(f"Empty content for '{task_name}'.")
                    return f"Error: Empty content for {task_name}."
                return content
            except asyncio.TimeoutError:
                log.error(f"Timeout for '{task_name}' after {timeout}s.")
                return f"Error: Timeout for {task_name} after {timeout}s."
            except Exception as e:
                log.error(f"Failed '{task_name}': {e}")
                return f"Error: LLM failed for {task_name}. Details: {str(e)}"

    def _parse_plan(self, response: str) -> List[Tuple[str, str]]:
        plan = []
        lines = response.strip().split("\n")
        for line in lines:
            if line.strip().startswith("- "):
                parts = line.strip()[2:].split(": ", 1)
                if len(parts) == 2:
                    plan.append((parts[0].strip(), parts[1].strip()))
        if not plan and response.strip().startswith("[") and response.strip().endswith("]"):
            try:
                parsed_plan = ast.literal_eval(response)
                if isinstance(parsed_plan, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in parsed_plan):
                    plan = [(str(item[0]), str(item[1])) for item in parsed_plan]
            except Exception as e:
                log.warning(f"AST parsing failed: {e}")
        if not plan:
            log.warning("Plan parsing failed; falling back to default.")
        return plan or self._fallback_plan("Unknown Query")

    def _fallback_plan(self, query: str) -> List[Tuple[str, str]]:
        log.warning("Using fallback plan.")
        return [
            ("Introduction", f"Introduce the topic: {query}"),
            ("Analysis", "Discuss findings related to the query."),
            ("Conclusion", "Summarize key points."),
            ("References", "Placeholder for sources.")
        ]

    def _format_retrieved_results_as_single_string(self, retrieved_results: List[RetrievalResult]) -> str:
        if not retrieved_results:
            return "No retrieved information available."
        formatted = ["Retrieved Information Context:\n"]
        for i, r in enumerate(retrieved_results):
            ref_str = json.dumps(r.reference) if r.reference else "N/A"
            text_snippet = r.text or ""
            formatted.append(f"<Document {i+1}>\nScore: {getattr(r, 'score', 'N/A')}\nReference: {ref_str}\nText:\n{text_snippet}\n</Document {i+1}>")
        return "\n\n".join(formatted)

    async def generate_plan(self, query: str, sub_queries: List[str], history_context: str) -> List[Tuple[str, str]]:
        prompt = PLANNING_PROMPT.format(
            query=query,
            sub_queries=", ".join(sub_queries) if sub_queries else "None",
            history_context=history_context[-30000:] if history_context else "None"
        )
        response = await self._call_llm(self.highllm, prompt, "plan")
        if response.startswith("Error:"):
            log.error(f"Plan generation failed: {response}")
            return self._fallback_plan(query)
        return self._parse_plan(response)

    async def map_sub_queries(self, section: str, description: str, sub_queries: List[str]) -> List[str]:
        if not sub_queries:
            return []
        prompt = MAPPING_PROMPT.format(section=section, description=description, sub_queries=", ".join(sub_queries))
        response = await self._call_llm(self.lightllm, prompt, "map")
        if response.startswith("Error:") or response.strip().lower() == "none":
            fallback_prompt = FALLBACK_MAPPING_PROMPT.format(section=section, description=description, sub_queries=", ".join(sub_queries))
            response = await self._call_llm(self.lightllm, fallback_prompt, "map_fallback")
            if response.startswith("Error:") or response.strip().lower() == "none":
                return []
        return [line.strip()[2:] for line in response.split("\n") if line.strip().startswith("- ") and line.strip()[2:] in sub_queries]

    async def draft_section(self, section: str, description: str, relevant_sub_queries: List[str], all_retrieved_info: str, plan_summary: str, original_query: str) -> Tuple[str, str]:
        prompt = DRAFT_SECTION_PROMPT.format(
            section=section, plan=description,
            relevant_sub_queries=", ".join(relevant_sub_queries) if relevant_sub_queries else "None",
            retrieved_info=all_retrieved_info, plan_summary=plan_summary, original_query=original_query
        )
        content = await self._call_llm(self.highllm, prompt, f"draft_{section}")
        if content.startswith("Error:"):
            return section, f"## {section}\n\n[Error: {content}]"
        return section, content

    async def polish_transition(self, first_section_end: str, second_section_start: str) -> str:
        prompt = POLISH_TRANSITION_PROMPT.format(
            first_section_end=first_section_end,
            second_section_start=second_section_start,
            n_sentences=self.transition_sentences
        )
        polished_transition = await self._call_llm(self.highllm, prompt, "polish")
        if polished_transition.startswith("Error:"):
            log.error(f"Transition polishing failed: {polished_transition}")
            return ""  # Return empty string to skip transition if polishing fails
        return polished_transition

    async def generate_response(self, query: str, retrieved_results: List[RetrievalResult], sub_queries: List[str], history_context: str) -> str:
        log.color_print("<think> Starting report generation... </think>\n", style="bold blue")

        # Step 1: Plan and format retrieved info
        log.color_print("<think> Planning and formatting... </think>\n", style="bold green")
        plan_task = asyncio.create_task(self.generate_plan(query, sub_queries, history_context))
        format_task = to_thread(self._format_retrieved_results_as_single_string, retrieved_results)
        plan, all_retrieved_info = await asyncio.gather(plan_task, format_task)
        if not plan:
            log.color_print("<think> Planning failed. Using fallback. </think>\n", style="bold red")
            plan = self._fallback_plan(query)
        plan_summary = "\n".join([f"- {section}: {desc}" for section, desc in plan])
        log.color_print("<think> Planning and formatting complete. </think>\n", style="bold green")

        # Step 2: Map sub-queries
        log.color_print("<think> Mapping sub-queries... </think>\n", style="bold green")
        map_tasks = [self.map_sub_queries(section, desc, sub_queries) for section, desc in plan]
        sub_query_mapping = dict(zip([section for section, _ in plan], await asyncio.gather(*map_tasks)))
        log.color_print("<think> Sub-query mapping complete. </think>\n", style="bold green")

        # Step 3: Draft sections
        log.color_print("<think> Drafting sections... </think>\n", style="bold green")
        draft_tasks = [
            self.draft_section(section, desc, sub_query_mapping.get(section, []), all_retrieved_info, plan_summary, query)
            for section, desc in plan
        ]
        drafted_sections = await asyncio.gather(*draft_tasks)
        section_dict = dict(drafted_sections)
        log.color_print("<think> Section drafting complete. </think>\n", style="bold green")

        # Step 4: Polish transitions and assemble
        log.color_print("<think> Polishing transitions... </think>\n", style="bold green")
        final_report = await self.polish_transitions(section_dict, [section for section, _ in plan])
        if not final_report:
            log.color_print("<think> Failed to assemble final report. </think>\n", style="bold red")
            return "Error: Failed to assemble the final report."

        log.color_print("<think> Report generation complete. </think>\n", style="bold blue")
        return final_report

    async def polish_transitions(self, section_dict: Dict[str, str], section_order: List[str]) -> str:
        if len(section_order) < 2:
            return section_dict.get(section_order[0], "") if section_order else ""

        polished_sections = []
        transition_tasks = []

        # Generate transitions for each pair
        for i in range(len(section_order) - 1):
            first_section = section_dict[section_order[i]]
            second_section = section_dict[section_order[i + 1]]

            first_sentences = self._extract_sentences(first_section, -self.transition_sentences)
            second_sentences = self._extract_sentences(second_section, self.transition_sentences)

            transition_tasks.append(self.polish_transition(first_sentences, second_sentences))

        polished_transitions = await asyncio.gather(*transition_tasks)

        # Assemble the report
        for i, section_title in enumerate(section_order):
            section_content = section_dict[section_title]
            if i < len(section_order) - 1:  # Not the last section
                transition = polished_transitions[i]
                if transition:
                    polished_sections.append(section_content + "\n\n" + transition)
                else:
                    polished_sections.append(section_content)
            else:  # Last section, no transition appended
                polished_sections.append(section_content)

        return "\n\n".join(polished_sections)

    def _extract_sentences(self, text: str, n: int) -> str:
        """Extract n sentences from start (if n > 0) or end (if n < 0), handling edge cases."""
        if not text.strip():
            return ""
        sentences = re.split(r'(?<=[.!?])\s+(?=(?:[^A-Z0-9]*[A-Z])|\d+\s)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        if n > 0:
            return ' '.join(sentences[:n]) if len(sentences) >= n else ' '.join(sentences)
        else:
            return ' '.join(sentences[n:]) if len(sentences) >= abs(n) else ' '.join(sentences)