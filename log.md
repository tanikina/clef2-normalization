## Log file documenting the progress on the Shared Task

### 07.03.25 Discussion and task distribution
Participants: Natalia, Ivan, Sebastian, Ravi, Tatiana

**Issues**: sometimes a single post may contain multiple claims, it is not clear what is the best strategy for claim extraction in this case (this needs some evaluation on the development data).

**Demonstrations**: based on cosine similarity to the available claims, also the prompt should contain a definition of normalized claims, e.g. from [(Sundriyal et al., 2023)](https://aclanthology.org/2023.findings-emnlp.439/).

**Open questions**: we are not sure yet whether we want to try fine-tuning language models with adapters or whether out-of-the-box inference with large enough models will already work fine.

TODO: come up with the team name and register for the task.

**Task distribution**:
  - Natalia: pre-processing, demostration selection
  - Ivan: prepare some code for inference  
  - Tatiana: prepare some code for tuning adapters and later for post-checking  
  - Ravi: post-processing and best claim selection (will start working on this later)  
  - Sebastian: has experience with summarization with LLMs, we will need some expertise on this

**Next steps**:
1. Run the baseline code, have a look at the results for known languages, identify the problems.
2. Come up with the "normalization strategy": what are the best prompts, do we need to translate the instruction in the target language etc?
3. Collect information about multilingual models, start testing them with out-of-the-box inference (when the code is ready).
