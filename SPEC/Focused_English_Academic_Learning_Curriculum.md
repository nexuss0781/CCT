# Focused-English Academic Learning Curriculum

## Overview

This curriculum defines the learning path for developing CCT from a basic English sequence learner into a conversational English model that can understand ambiguity, ask for clarification, maintain context, and repair misunderstandings.

The curriculum is progressive. Each module introduces a new capability only after the preceding capability has been learned sufficiently. The model should not be treated as conversational before it has developed the underlying abilities required for conversation.

The learning progression is:

- **Symbols and words** — learn the stable units of English.
- **Sentences** — learn how words form grammatical statements.
- **Meaning** — learn how sentences express events, entities, relations, and intentions.
- **Paragraphs** — learn how several sentences maintain a topic and develop an idea.
- **Comprehension** — learn to interpret supplied English and answer from its meaning.
- **Instructions** — learn to respond according to explicit requirements.
- **Ambiguity** — learn when a request has more than one reasonable interpretation.
- **Clarification** — learn to ask the question that resolves the uncertainty.
- **Conversation** — learn to maintain context, accept correction, and repair misunderstandings.
- **Transfer and conversational maturity** — apply the learned abilities to new but bounded situations.

Each module contains:

- **Goal:** the main learning purpose.
- **Competencies:** the abilities the model must acquire.
- **Submodules:** the smaller lessons that build the competency.
- **Expected outcome:** what the model should be able to do before progressing.

The curriculum is about **what the model must learn**. Training configuration, datasets, checkpointing, resource measurement, and implementation gates belong in separate engineering documents.

---

# Module 1 — English Symbols, Words, and Boundaries

## Goal

Enable the model to represent and reproduce the basic visible structure of English text reliably.

## Competencies

- Recognize common English letters, symbols, digits, and punctuation.
- Distinguish spaces from characters that belong inside words.
- Preserve the boundaries between words.
- Recognize common short and long English words as stable sequences.
- Reproduce ordinary punctuation without collapsing or duplicating it.
- Avoid immediate repetition of a single character, token, or short sequence.
- Continue a short English fragment without producing malformed or unusable text.

## Submodules

### 1.1 Character and symbol awareness

- Uppercase and lowercase letters.
- Digits and common symbols.
- Periods, commas, question marks, exclamation marks, colons, and semicolons.
- Apostrophes, quotation marks, parentheses, and hyphens.
- Recognition of symbols at the beginning, middle, and end of text.

### 1.2 Whitespace and word boundaries

- Single spaces between words.
- Spaces after punctuation.
- Words at the beginning and end of a sentence.
- Boundaries around quotation marks and parentheses.
- Boundaries in contractions such as `don't` and `can't`.

### 1.3 Common word patterns

- Short function words such as `the`, `a`, `is`, `of`, `to`, and `and`.
- Common prefixes and suffixes.
- Singular and plural word forms.
- Repeated letters and common spelling patterns.
- Frequent educational and everyday vocabulary.

### 1.4 Stable short continuation

- Continue a short word or phrase.
- Complete a familiar local pattern.
- Preserve the prompt’s punctuation style.
- Avoid empty, immediately terminated, or endlessly repetitive output.

## Expected outcome

The model produces readable short English continuations with stable symbols, visible word boundaries, and ordinary punctuation. This module does not require fluent sentences or conversation.

---

# Module 2 — Words, Vocabulary, and Lexical Relations

## Goal

Develop a usable basic English vocabulary and learn how words relate to one another in local context.

## Competencies

- Recognize common nouns, verbs, adjectives, adverbs, pronouns, and function words.
- Distinguish common word meanings from their local context.
- Recognize singular and plural forms.
- Recognize common tense and derivational forms.
- Identify simple synonyms, opposites, and category relationships.
- Select words that fit the surrounding sentence.
- Avoid replacing a required word with an unrelated word.

## Submodules

### 2.1 Core vocabulary

- People, objects, places, actions, qualities, and time.
- Everyday activities and educational topics.
- Common abstract words such as `reason`, `change`, `result`, and `problem`.
- High-frequency connectors and grammatical words.

### 2.2 Word classes

- Nouns and noun phrases.
- Verbs and verb phrases.
- Adjectives and adjective phrases.
- Adverbs and adverbial phrases.
- Pronouns, determiners, conjunctions, and prepositions.

### 2.3 Morphological patterns

- Singular and plural nouns.
- Present, past, and participle verb forms.
- Comparative and superlative adjectives.
- Common prefixes and suffixes.
- Negation forms and contractions.

### 2.4 Lexical relations

- Synonyms and near-synonyms.
- Antonyms.
- Part–whole relations.
- Category–member relations.
- Agent–action and object–action relations.
- Cause–effect vocabulary.

### 2.5 Contextual word selection

- Selecting a word that fits the subject.
- Selecting a word that fits the verb.
- Selecting a word that fits the intended meaning.
- Distinguishing common words with different uses.
- Avoiding locally plausible but contextually unrelated substitutions.

## Expected outcome

The model uses a growing English vocabulary in locally appropriate ways and can select words that fit simple meanings and relations.

---

# Module 3 — Sentence Formation and Local Grammar

## Goal

Enable the model to form and complete grammatical English sentences.

## Competencies

- Complete a sentence in a way that fits its beginning.
- Maintain subject–verb agreement.
- Maintain consistent tense.
- Use articles and determiners appropriately.
- Use pronouns with clear local reference.
- Form basic affirmative, negative, and interrogative sentences.
- Use common prepositions and conjunctions correctly.
- End sentences with appropriate punctuation.

## Submodules

### 3.1 Sentence constituents

- Subjects and predicates.
- Noun phrases and verb phrases.
- Objects and complements.
- Modifiers and simple adjuncts.
- Basic sentence patterns.

### 3.2 Agreement

- Singular subject with singular verb.
- Plural subject with plural verb.
- Agreement with compound subjects.
- Agreement with pronouns.
- Agreement across short inserted phrases.

### 3.3 Tense and aspect

- Present events.
- Past events.
- Future events.
- Ongoing actions.
- Completed actions.
- Consistent tense across adjacent sentences.

### 3.4 Articles and determiners

- `a` and `an`.
- `the`.
- Zero article in common contexts.
- Demonstratives such as `this`, `that`, `these`, and `those`.
- Possessive determiners.

### 3.5 Sentence types

- Statements.
- Questions.
- Commands.
- Exclamations.
- Negation.
- Short answers.

### 3.6 Coordination and subordination

- `and`, `but`, and `or`.
- `because`, `so`, and `although`.
- Time relations such as `when`, `before`, and `after`.
- Simple conditional forms.
- Cause, contrast, and consequence.

## Expected outcome

The model completes unseen sentences with locally grammatical structure, preserves tense and agreement, and produces understandable statements and questions.

---

# Module 4 — Sentence Meaning and Semantic Composition

## Goal

Enable the model to construct and interpret the meaning expressed by complete sentences.

## Competencies

- Identify who or what a sentence concerns.
- Identify actions, states, properties, and events.
- Connect actions to their participants.
- Distinguish time, location, manner, and quantity.
- Understand simple negation.
- Understand simple questions and requested information.
- Preserve the meaning of a sentence when rephrasing it.
- Distinguish related but non-equivalent statements.

## Submodules

### 4.1 Entities and participants

- People and groups.
- Objects and locations.
- Animals and natural entities.
- Abstract entities and events.
- Agent, patient, recipient, and experiencer roles.

### 4.2 Events and states

- Actions.
- Conditions.
- Changes.
- Possession.
- Existence.
- Perception and knowledge.

### 4.3 Time and order

- Before and after.
- Earlier and later.
- Duration.
- Frequency.
- Present, past, and future reference.
- Simple event sequences.

### 4.4 Negation and contrast

- `not` and negative auxiliaries.
- Absence and refusal.
- Contrast with `but` and `although`.
- Distinguishing positive from negative claims.

### 4.5 Paraphrase and semantic preservation

- Restating a sentence in simpler language.
- Replacing words with appropriate synonyms.
- Preserving the original participants and event.
- Avoiding added facts.
- Distinguishing a paraphrase from a contradiction.

## Expected outcome

The model produces sentences whose basic meaning is coherent and can preserve simple semantic relations when completing or rephrasing text.

---

# Module 5 — Paragraph Coherence and Topic Development

## Goal

Enable the model to maintain a topic and develop a connected idea across multiple sentences.

## Competencies

- Continue a paragraph about the same topic.
- Preserve the identity of people, objects, and places.
- Maintain consistent time and event order.
- Connect sentences through cause, contrast, example, and consequence.
- Avoid abrupt and unrelated topic changes.
- Introduce and develop a simple explanation.
- Conclude a short paragraph appropriately.

## Submodules

### 5.1 Topic maintenance

- Identifying the central topic.
- Reusing relevant vocabulary.
- Avoiding unrelated topic shifts.
- Distinguishing the main topic from an incidental detail.

### 5.2 Entity continuity

- Keeping names and descriptions consistent.
- Tracking a person or object across sentences.
- Preserving pronoun reference.
- Avoiding entity substitution or disappearance.

### 5.3 Discourse relations

- Cause and effect.
- Problem and solution.
- Claim and evidence.
- Question and answer.
- General statement and example.
- Contrast and comparison.

### 5.4 Temporal and causal order

- Maintaining event sequence.
- Preserving before-and-after relationships.
- Avoiding contradictions in time.
- Connecting causes to appropriate results.

### 5.5 Paragraph closure

- Summarizing a short development.
- Completing an explanation.
- Reaching a simple conclusion.
- Avoiding abrupt termination or unrelated continuation.

## Expected outcome

The model produces short coherent paragraphs that stay on topic, preserve entities, and develop a connected sequence of ideas.

---

# Module 6 — Reading Comprehension and Meaning Retrieval

## Goal

Enable the model to read supplied English text and answer questions using the information expressed in that text.

## Competencies

- Identify explicit information in a passage.
- Connect a question to the relevant sentence or passage section.
- Answer with the correct entity, action, property, or event.
- Combine information from more than one sentence.
- Distinguish relevant from irrelevant details.
- Recognize when the passage does not contain the answer.
- Avoid presenting unsupported information as if it came from the passage.

## Submodules

### 6.1 Explicit information retrieval

- Finding names, dates, places, quantities, and actions.
- Matching a question to a direct statement.
- Returning concise answers.

### 6.2 Local inference

- Combining two nearby statements.
- Recognizing simple cause and effect.
- Resolving references within the passage.
- Inferring a straightforward unstated relation.

### 6.3 Relevance selection

- Selecting the sentence that answers the question.
- Ignoring distractor information.
- Distinguishing the main fact from supporting detail.

### 6.4 Unanswerable questions

- Recognizing missing information.
- Stating that the passage does not provide the answer.
- Avoiding confident invention.
- Distinguishing unknown information from negative information.

### 6.5 Answer formulation

- Short factual answers.
- Evidence-based explanations.
- Answering the question asked rather than a related question.
- Preserving names, quantities, and relations accurately.

## Expected outcome

The model answers questions from supplied text accurately and can indicate when the text does not support an answer.

---

# Module 7 — Instruction Understanding and Task Completion

## Goal

Enable the model to understand explicit instructions and produce responses that satisfy the requested task and format.

## Competencies

- Identify what the user is asking for.
- Separate the task from background information.
- Follow one-step and multi-step instructions.
- Preserve requested order and format.
- Include required information.
- Exclude prohibited or irrelevant information.
- Recognize impossible, contradictory, or incomplete instructions.
- State limitations instead of pretending to complete an impossible task.

## Submodules

### 7.1 Instruction identification

- Action requested by the user.
- Object of the action.
- Constraints on the response.
- Expected output form.
- Desired level of detail.

### 7.2 Single-step tasks

- Summarize.
- Classify.
- Extract.
- Rewrite.
- Explain.
- Compare.
- Transform a supplied text.

### 7.3 Multi-step tasks

- Ordering subtasks.
- Preserving intermediate information.
- Completing all requested parts.
- Checking that no subtask was omitted.

### 7.4 Format and constraint following

- Lists.
- Tables.
- Short answers.
- Specified lengths.
- Required headings.
- Exact or bounded formats.

### 7.5 Limits and conflicts

- Missing required information.
- Contradictory instructions.
- Impossible requirements.
- Requests outside the available context.
- Explaining the limitation and asking for the necessary correction.

## Expected outcome

The model follows clear instructions accurately, produces the requested response form, and acknowledges limitations when the task cannot be completed as stated.

---

# Module 8 — Ambiguity Recognition

## Goal

Enable the model to recognize when a prompt has multiple reasonable interpretations or lacks information necessary for a reliable answer.

## Competencies

- Distinguish clear prompts from ambiguous prompts.
- Identify the specific source of ambiguity.
- Recognize missing parameters.
- Recognize unclear references.
- Recognize ambiguous scope or intention.
- Recognize conflicting requirements.
- Distinguish genuine ambiguity from merely uncommon wording.
- Avoid silently selecting an interpretation when the difference matters.

## Submodules

### 8.1 Lexical ambiguity

- Words with multiple meanings.
- Domain-dependent terminology.
- Common words used technically.
- Choosing between meanings from context.

### 8.2 Referential ambiguity

- Unclear pronoun reference.
- Multiple possible antecedents.
- Unclear names or descriptions.
- Ambiguous references across turns.

### 8.3 Scope ambiguity

- Unclear quantity or group membership.
- Unclear modifier attachment.
- Unclear order of operations.
- Unclear boundary of the requested task.

### 8.4 Missing-parameter ambiguity

- Missing time, location, audience, format, or quantity.
- Missing definition of success.
- Missing relevant background information.
- Missing choice among materially different alternatives.

### 8.5 Contradictory and underspecified requests

- Requirements that cannot all be satisfied.
- Requests with insufficient detail.
- Requests that depend on an unstated preference.
- Requests where a guess could lead to a materially different result.

### 8.6 Ambiguity versus clarity

- Recognizing when context already resolves the ambiguity.
- Avoiding unnecessary clarification.
- Distinguishing a difficult question from an ambiguous question.
- Distinguishing lack of knowledge from lack of user specification.

## Expected outcome

The model reliably identifies when a request requires clarification and can explain what information is missing or what interpretations are possible.

---

# Module 9 — Clarification and Intent Alignment

## Goal

Enable the model to ask concise, useful questions that resolve ambiguity before attempting the final task.

## Competencies

- Ask for the missing information that matters most.
- Present the relevant alternatives clearly.
- Avoid asking questions that do not affect the answer.
- Ask one focused question when one question is sufficient.
- Ask a small ordered set of questions when several details are necessary.
- Preserve the user’s original objective while clarifying it.
- Proceed directly when the request is already clear.

## Submodules

### 9.1 Clarification question selection

- Identify the decision that blocks a reliable answer.
- Ask about the highest-impact missing detail first.
- Avoid requesting information that can be inferred safely.

### 9.2 Alternative presentation

- State the possible interpretations.
- Use neutral wording.
- Avoid forcing the user toward one option.
- Explain the practical difference between alternatives when useful.

### 9.3 Minimal clarification

- Prefer the fewest questions needed.
- Avoid repetitive or circular questions.
- Avoid turning a simple request into an interview.

### 9.4 Clarification and continuation

- Use the user’s answer to resolve the original request.
- Preserve the clarified intention.
- Do not return to the original ambiguity after it is resolved.

### 9.5 Direct response control

- Answer clear requests directly.
- Clarify only when ambiguity is material.
- State assumptions when a safe, low-impact assumption is appropriate.
- Ask before acting when an assumption could change the outcome significantly.

## Expected outcome

The model asks relevant clarification questions when necessary and answers directly when the user’s intention is sufficiently clear.

---

# Module 10 — Conversational Context and Continuity

## Goal

Enable the model to maintain relevant information across multiple turns and use the conversation history appropriately.

## Competencies

- Retain facts introduced earlier in the conversation.
- Track the current topic and user objective.
- Resolve references such as `it`, `that`, `the earlier one`, and `this approach`.
- Distinguish current instructions from outdated instructions.
- Use the latest user preference or correction.
- Avoid repeating questions that have already been answered.
- Ignore irrelevant earlier information when it no longer applies.

## Submodules

### 10.1 Conversation state

- Current topic.
- User objective.
- Established facts.
- Open questions.
- Completed actions.
- Pending decisions.

### 10.2 Reference across turns

- Pronouns.
- Ellipsis and omitted information.
- Repeated descriptions.
- References to previous alternatives.
- References to earlier answers.

### 10.3 Topic continuity

- Continue the active subject.
- Recognize a legitimate topic change.
- Distinguish a follow-up from a new request.
- Avoid unrelated response shifts.

### 10.4 Instruction updates

- New instruction replacing an old instruction.
- User correction of a previous preference.
- Addition of a new constraint.
- Removal of an earlier constraint.
- Confirming which instruction is currently active.

### 10.5 Context relevance

- Retain useful history.
- Ignore irrelevant history.
- Avoid copying old details into a new answer without reason.
- Preserve the user’s intended level of detail.

## Expected outcome

The model maintains a coherent conversational thread and uses relevant history without becoming confused by outdated or irrelevant context.

---

# Module 11 — Misunderstanding Detection and Conversational Repair

## Goal

Enable the model to recognize when its interpretation or response was wrong and repair the interaction constructively.

## Competencies

- Recognize explicit user correction.
- Detect disagreement between its response and the user’s intended request.
- Acknowledge misunderstanding without becoming defensive.
- State the corrected interpretation.
- Ask a clarification question when the correction remains incomplete.
- Produce a revised answer based on the corrected intent.
- Preserve useful earlier context while discarding the mistaken assumption.

## Submodules

### 11.1 Correction recognition

- Direct correction.
- Partial correction.
- User rejection of an assumption.
- User restatement of the intended request.

### 11.2 Error acknowledgment

- Identify what was misunderstood.
- Avoid pretending the previous response was correct.
- Avoid blaming the user for the misunderstanding.
- Use concise acknowledgment.

### 11.3 Interpretation repair

- Restate the corrected intent.
- Confirm the interpretation when necessary.
- Separate corrected information from unchanged context.

### 11.4 Response revision

- Answer the corrected request.
- Remove consequences of the earlier wrong assumption.
- Preserve valid parts of the earlier work.
- Avoid repeating the same error.

### 11.5 Contradiction handling

- Recognize conflicting statements.
- Identify which statement is newer or authoritative in context.
- Ask for resolution when the conflict cannot be resolved safely.
- Explain the consequence of each possible interpretation.

## Expected outcome

The model can recover from misunderstandings, update its interpretation, and continue the conversation without resetting or repeating the original error.

---

# Module 12 — Conversational Pragmatics and Natural English Interaction

## Goal

Enable the model to use English appropriately for real conversational purposes rather than producing only grammatically correct sentences.

## Competencies

- Distinguish literal wording from conversational intention.
- Recognize requests, suggestions, refusals, explanations, corrections, and confirmations.
- Match tone and detail to the user’s context.
- Be concise when the user wants a direct answer.
- Explain carefully when the user needs reasoning.
- Express uncertainty without becoming unhelpful.
- Avoid unnecessary confidence, evasion, or irrelevant verbosity.

## Submodules

### 12.1 Communicative intention

- Asking.
- Telling.
- Requesting.
- Suggesting.
- Correcting.
- Confirming.
- Refusing.
- Negotiating.

### 12.2 Context-sensitive tone

- Neutral professional tone.
- Friendly explanatory tone.
- Concise operational tone.
- Respectful disagreement.
- Sensitive correction.
- Appropriate formality.

### 12.3 Relevance and concision

- Answer the actual question.
- Prioritize the most useful information.
- Avoid unnecessary repetition.
- Expand only when explanation is useful.

### 12.4 Uncertainty and epistemic discipline

- Distinguish known information from inference.
- State uncertainty when evidence is incomplete.
- Avoid fabricated certainty.
- Ask for information when confidence depends on missing context.

### 12.5 Social interaction patterns

- Greeting and closing.
- Confirmation of shared understanding.
- Polite disagreement.
- Repair after confusion.
- Respectful handling of user preferences.

## Expected outcome

The model communicates naturally enough for focused English conversation, while remaining relevant, clear, appropriately cautious, and responsive to the user’s intention.

---

# Module 13 — Bounded Transfer and New-Domain Application

## Goal

Enable the model to apply its learned English and conversational abilities to new topics and situations within a declared scope.

## Competencies

- Apply sentence and paragraph abilities to unfamiliar subjects.
- Apply comprehension skills to new passage styles.
- Follow instructions in new task formats.
- Recognize ambiguity in new domains.
- Ask appropriate clarification questions in unfamiliar contexts.
- Maintain conversation when new terminology appears.
- State when a domain requires knowledge or context not available to the model.

## Submodules

### 13.1 Topic transfer

- Everyday topics.
- Educational topics.
- Practical planning topics.
- Descriptive and explanatory topics.
- New vocabulary within familiar structures.

### 13.2 Format transfer

- Questions.
- Explanations.
- Summaries.
- Comparisons.
- Plans.
- Structured responses.

### 13.3 Ambiguity transfer

- Ambiguous requests in new domains.
- Domain-specific terms with multiple meanings.
- Missing constraints in unfamiliar tasks.
- Clarification without unnecessary refusal.

### 13.4 Conversation transfer

- Follow-up questions in new subjects.
- Corrections involving unfamiliar terms.
- Topic shifts.
- Multi-step objectives.

### 13.5 Boundary recognition

- Recognizing the limits of the learned scope.
- Asking for definitions or context.
- Avoiding unsupported claims.
- Preserving useful communication even when knowledge is incomplete.

## Expected outcome

The model applies its learned language and conversational capabilities to new but bounded situations without losing earlier competencies or pretending to know what it cannot establish.

---

# Module 14 — Integrated Conversational Mastery

## Goal

Integrate the preceding abilities into a coherent conversational English model.

## Competencies

- Understand the user’s words and intended task.
- Form grammatically sound and semantically relevant responses.
- Maintain context across turns.
- Detect ambiguity before making a consequential assumption.
- Ask concise clarification questions.
- Follow the clarified intention.
- Recognize and repair misunderstandings.
- Adapt response length, tone, and structure.
- Transfer these abilities to new but bounded topics.

## Submodules

### 14.1 Integrated single-turn conversation

- Clear factual requests.
- Explanations.
- Comparisons.
- Summaries.
- Corrections.
- Requests with explicit constraints.

### 14.2 Integrated ambiguous conversation

- Ambiguous initial requests.
- Missing information.
- Multiple possible interpretations.
- Clarification followed by task completion.

### 14.3 Integrated multi-turn conversation

- Context retention.
- User preference changes.
- Follow-up questions.
- Corrections.
- Topic changes.
- Repair after misunderstanding.

### 14.4 Integrated bounded transfer

- New domains.
- New prompt formats.
- Unfamiliar terminology.
- Incomplete information.
- Mixed conversational objectives.

### 14.5 Conversational quality

- Relevance.
- Coherence.
- Natural English.
- Appropriate uncertainty.
- Appropriate clarification.
- Respectful interaction.
- Stable behavior across repeated conversations.

## Expected outcome

The model demonstrates a coherent, native-English-oriented conversational capability within the declared scope. It understands ordinary requests, forms useful responses, detects meaningful ambiguity, clarifies when required, maintains context, and repairs misunderstandings.

---

# Module 15 — Final Learning Objective

## Goal

Reach the intended endpoint of the curriculum: an efficient, trainable CCT language engine that can participate in English conversation with reliable ambiguity understanding and repair within its tested scope.

## Final competencies

- Stable English representation.
- Usable vocabulary and lexical relations.
- Sentence completion and local grammar.
- Sentence-level semantic composition.
- Paragraph coherence and topic persistence.
- Reading comprehension and answer targeting.
- Instruction understanding and task completion.
- Ambiguity recognition.
- Clarification and intent alignment.
- Conversational context retention.
- Misunderstanding detection and repair.
- Natural English pragmatics.
- Bounded transfer to new topics and formats.

## Final learning outcome

The model should be able to:

- Understand what a user has said.
- Determine what the user is trying to accomplish.
- Recognize when the request is clear.
- Recognize when the request is ambiguous or underspecified.
- Ask the smallest useful clarification question.
- Complete the clarified task in natural English.
- Maintain relevant conversational context.
- Accept correction and revise its interpretation.
- Repair misunderstandings without repeating the original error.
- Express uncertainty when the available information is insufficient.
- Apply the same abilities to new but bounded conversational situations.

This is the endpoint of the learning path. It is a **measured language-learning objective**, not an unsupported claim of general intelligence or unrestricted human-level ability.
