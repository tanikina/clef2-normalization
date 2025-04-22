## Log file contains info about experiments with Qwen used for inference for the task of normalizing original and clean claims  

### Cleaning the data
The data (train, dev and test for all the languages) were cleaned using regular expressions as follows:
 1) The emojis and excessive punctuation were removed; whitespaces / newlines normalized; duplicate sentences (if could be split by .?!) were removed.
 2) Meaningful tokens from the weblinks were extracted, e.g., 'https://www.technocracy.news/blaylock-face-masks-pose-serious-risks-to-the-healthy/' was converted to 'https://www.technocracy.news/ blaylock face masks pose serious risks to the healthy'.
 3) Meaningful tokens from the hashtags were extracted, e.g., '#MasksDoNotWork' was converted into 'masks do not work'.

**Issues**: Note that by doing steps 2) and 3) it was impossible to completely avoid non-meaningful tokens in the output. In addition it is impossible to evaluate the quality of the outputs for languages one is not familiar with.

### Experiments with Qwen (inference only) with clean and original claims and different number of demonstrations

**Demonstrations**: The idea is to pick out semantically similar demonstrations from the _train_ partition(s) for the claims in the _dev_ and _test_ partitions. We calculate semantic similarity using Sentence BERT (Reimers, N. and Gurevych, I., 2020) with the multilingual _sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2_ model (supports 50 languages). For each instrance in _dev_ and _test_ we calculate two types of the most similar demonstrations: 1) demonstration from the same language and 2) global (i.e. multilingual) demonstrations. Global demonstrations can be used for claims in languages that do not have the corresponding _train_ partitions. The information about semantically similar demonstrations is stored as _.jsonlines_ files in the _./clean_data/_ and _./data/_ folders.

**Note**: For clean instances we take clean demonstrations, for original ones - original demonstrations. We also experiment with random demonstrations.

**Results** on the _dev_ partitions (avg. of 3 different runs)

| Setting | ara | deu | eng | fra | hi | mr | msa | pa | pol | por | spa | ta | tha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B-instruct <br>0 orig shots (baseline) | **0.352** | 0.138 | 0.231 | 0.246 | 0.205 | 0.264 | 0.197 | 0.247 | 0.148 | 0.288 | 0.270 | 0.329 | 0.079 |
| Qwen2.5-7B-instruct <br>2 clean most sim shots <br>same lang | 0.140 | **0.278** | 0.428 | 0.349 | 0.233 | 0.279 | 0.339 | 0.278 | 0.254 | 0.438 | 0.374 | 0.421 | 0.155 |
| Qwen2.5-7B-instruct <br>3 clean most sim shots <br>same lang | 0.140 | 0.267 | 0.431 | 0.353 | **0.248** | 0.269 | 0.335 | 0.282 | 0.255 | 0.444 | 0.389 | **0.425** | 0.174 |
| Qwen2.5-7B-instruct <br>2 clean random shots <br>same lang | 0.140 | 0.171 | 0.275 | 0.276 | 0.190 | 0.293 | 0.214 | 0.253 | 0.170 | 0.268 | 0.227 | 0.406 | 0.061 |
| Qwen2.5-7B-instruct <br>1 orig most sim shots <br>same lang | 0.326 | 0.214 | 0.418 | 0.386 | 0.224 | 0.273 | 0.370 | 0.212 | **0.288** | 0.387 | 0.369 | 0.356 | 0.184 |
| Qwen2.5-7B-instruct <br>2 orig most sim shots <br>same lang | 0.319 | 0.222 | 0.436 | 0.382 | 0.220 | 0.320 | 0.380 | 0.276 | 0.282 | **0.459** | 0.412 | 0.371 | **0.189** |
| Qwen2.5-7B-instruct <br>3 orig most sim shots <br>same lang | 0.330 | 0.237 | 0.439 | 0.391 | 0.247 | **0.331** | **0.385** | 0.302 | 0.282 | 0.451 | 0.409 | 0.411 | 0.166 |
| Qwen2.5-7B-instruct <br>3 orig most sim shots <br>multiling | 0.251 | 0.186 | 0.416 | 0.301 | 0.206 | 0.282 | 0.339 | **0.308** | 0.191 | 0.414 | 0.385 | 0.310 | 0.150 |
| Qwen2.5-7B-instruct <br>4 orig most sim shots <br>same lang | 0.334 | 0.230 | **0.442** | **0.404** | 0.244 | 0.292 | 0.378 | 0.287 | 0.267 | 0.453 | **0.417** | 0.412 | 0.139 |
| Gemma-3-27B-it <br>3 orig most sim shots <br>same lang | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. | 0. |


**Main takeaways**:
- Cleaning the posts doesn't bring any effects (except shorter inference time)
- Demonstrations are useful
- Original demonstrations are better than clean ones
- Most semantically similar demonstrations are better than random ones
- Number of demonstrations depends on the language
- Demonstrations in the same language are typically better than multilingual ones.