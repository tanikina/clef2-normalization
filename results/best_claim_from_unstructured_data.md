# Results : Monolingual Experiments

## Approach 1: 
We prompted LLMs to create a best normalized claim from the unstructured data with two examples from the train set as samples and evaluated on dev set

| model                 |   ara |   deu |   eng |   fra |    hi |    mr |   msa |    pa |   pol |   por |   spa |    ta |   tha |
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Llama-4-17b-instruct  |0.3506 |0.2290 |0.2828 |0.2958 |0.2500 |0.1431 |0.2861 |0.2275 |0.2662 |0.2857 |0.2993 |0.2461 |0.1890 |
| Llama-3.3-70B         |0.3335 |0.2293 |0.2863 |0.3221 |0.2528 |0.1724 |0.2810 |0.1835 |0.2459 |0.3276 |0.3044 |0.1718 |0.1362 |
| mistral-24B           |0.3472 |0.2097 |0.2928 |0.2952 |0.2278 |0.1521 |0.2584 |0.2939 |0.2691 |0.2899 |0.2940 |0.3212 |0.1603 |


## Approach 2: 
We prompted LLMs to summarize the unstructured data into a normalized claim, we gave two examples from the train set and evaluated on dev set

| model                 |   ara |   deu |   eng |   fra |    hi |    mr |   msa |    pa |   pol |   por |   spa |    ta |   tha |
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Llama-4-17b-instruct  |0.3478 |0.2181 |0.2699 |0.2887 |0.2652 |0.1321 |0.2626 |0.2231 |0.2751 |0.3088 |0.3100 |0.2485 |0.1730 |
| Llama-3.3-70B         |0.3414 |0.2420 |0.2887 |0.3238 |0.2522 |0.1816 |0.2735 |0.1825 |0.2503 |0.3267 |0.3071 |0.1577 |0.1546 |
| mistral-24B           |0.3410 |0.2183 |0.2989 |0.2894 |0.2358 |0.1576 |0.2527 |0.2982 |0.2526 |0.2960 |0.2958 |0.3393 |0.1399 |
