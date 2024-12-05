This folder contain code & data to replicate Jorge et al. paper (Benatallah, Boualem, and Fabio Casati. "Crowdsourcing Syntactically Diverse Paraphrases with Diversity-Aware Prompts and Workflows." Advanced Information Systems Engineering: 34th International Conference, CAiSE 2022, Leuven, Belgium, June 6–10, 2022, Proceedings. Vol. 13295. Springer Nature, 2022).


In this study we investigate whether GPT can replace crowdsourcing in the case of Crowdsourcing syntactically diverse paraphrases with diversity-aware prompts task.

We posit that by designig syntax-aware prompts we can effectively guide GPT towards and syntactic diversity. Is GPT generation better than crowds.


PSA.csv is the seed_dataset used by Jorge.



TED_plot_comparison_random.ipynb is same as TED_plot_comparison.ipynb however instead of selecting top-3 paraphrases per seed_utterances according to BERTscore we select them randomly. Here we can compare GPT towards best and worst paraphrases generated through crowdsourcing.



Based on the Tree Edit Distance (TED) scores between the two bootstrap dataset the one that we replicate (see bootstrap_round.ipynb i.e. Dataset1) and (Jorge one see TED_plot_comparison.ipynb i.e. Dataset 2). Dataset 1 appears to have a better syntactic diversity compared to Dataset 2. In Dataset 2, the median and mean TED scores for all three columns (p1, p2, and p3) are consistently lower than those in Dataset 1. This suggests that the paraphrases in Dataset 2 exhibit less syntactic variation, indicating a tighter syntactic structure overall. On the other hand, Dataset 1 shows higher median and mean TED scores, indicating a relatively broader range of syntactic patterns and potentially more diverse sentence structures.

[Dataset 1]:
p1_ted: Median: 4.0000. Mean: 4.1373.
p2_ted: Median: 5.0000. Mean: 4.5294.
p3_ted: Median: 5.0000. Mean: 4.2549.

[Dataset 2]:
p1_ted: Median: 2.0000. Mean: 2.7647.
p2_ted: Median: 4.0000. Mean: 3.3137.
p3_ted: Median: 3.0000. Mean: 3.2549.





Based on these scores, we can make the following observations:

    Median BERTScore Comparison:
        Dataset 2 has a higher median BERTScore (0.9601) compared to Dataset 1 (0.9222).
        A higher median BERTScore indicates that Dataset 2 has a higher semantic similarity with the reference text on average.

    Mean BERTScore Comparison:
        Dataset 2 has a higher mean BERTScore (0.9562) compared to Dataset 1 (0.9228).
        Similar to the median comparison, a higher mean BERTScore suggests that Dataset 2 has a higher overall semantic similarity with the reference text.

Based on these observations, we can conclude that according to BERTScore, Dataset 2 performs better in terms of semantics compared to Dataset 1. The higher median and mean BERTScores for Dataset 2 indicate that the generated text in Dataset 2 is more semantically similar to the reference text, on average, compared to Dataset 1.


Here's how you can interpret the standard deviation:

    Spread of Data: A larger standard deviation indicates that the data points are more spread out from the mean, suggesting greater variability or dispersion within the dataset. A smaller standard deviation suggests that the data points are closer to the mean, indicating less variability.

    Data Distribution: The standard deviation gives you a sense of the shape of the data distribution. In a normal distribution (bell curve), about 68% of the data falls within one standard deviation of the mean, about 95% falls within two standard deviations, and about 99.7% falls within three standard deviations.

    Comparison: When comparing different datasets or groups, the standard deviation helps you understand how consistent or variable the values are within each group. A smaller standard deviation suggests more consistent values, while a larger standard deviation suggests greater variability.
    
    
===================================================================================================
PINC DIV and TTR between bootstrap

[Dataset 1]:  Mean TTR: 0.5424  Mean PINC: 0.4471  DIV: 0.8121
[Dataset 2]: Mean TTR: 0.4252  Mean PINC: 0.2158  DIV: 0.3971

Type-Token Ratio (TTR):
The Type-Token Ratio (TTR) measures the diversity or richness of vocabulary in a given dataset. A higher TTR indicates a greater diversity of words being used. Looking at the results:
    Dataset 1 has a Mean TTR of 0.5424, suggesting a relatively higher diversity of vocabulary.
    Dataset 2 has a Mean TTR of 0.4252, indicating a lower diversity of vocabulary compared to Dataset 1.

Paraphrase In N-gram Changes (PINC):
The Paraphrase In N-gram Changes (PINC) score measures the degree of novelty of automatically generated paraphrases. A higher PINC score indicates that the paraphrases have more significant changes from the original text.
    Dataset 1 has a Mean PINC score of 0.4471, suggesting that the generated paraphrases exhibit a moderate level of novelty.
    Dataset 2 has a Mean PINC score of 0.2158, indicating that the generated paraphrases have relatively fewer significant changes from the original text compared to Dataset 1.

Diversity (DIV):
The DIV metric computes diversity at the corpus level by calculating n-gram changes between all pairs of utterances sharing the same intent. A higher DIV score indicates a greater diversity of paraphrases within each intent group.
    Dataset 1 has a DIV score of 0.8121, indicating a relatively higher diversity of paraphrases within intent groups.
    Dataset 2 has a DIV score of 0.3971, suggesting a lower diversity of paraphrases within intent groups compared to Dataset 1.

In summary:

Dataset 1 generally exhibits higher vocabulary diversity, more novelty in generated paraphrases, and greater diversity within intent groups compared to Dataset 2.
Dataset 2 has lower vocabulary diversity, less novelty in generated paraphrases, and lower diversity within intent groups compared to Dataset 1.

===================================================================================================
===================================================================================================
===================================================================================================
===================================================================================================
bootstrap_round (146,) paraphrases

Step 2 - Compute metrics
	Mean TTR: 0.5424 
	Mean PINC: 0.4471 
	DIV: 0.8121
	
Calculating the Mean and Median for the Complete Paraphrase TED Score List
	Paraphrases Tree Edit Distance: Median: Median: 5.0000. Mean: 4.3699.


Calculate the individual mean and median scores for each paraphrase column.
	BERT scores: Median: 0.9358. Mean: 0.9340.

	BERT scores: Median: 0.9225. Mean: 0.9239.

	BERT scores: Median: 0.9170. Mean: 0.9194.


Calculating the Mean and Median for the Complete Paraphrase BERT Score List
	paraphrases_bertscores: Median: 0.9222. Mean: 0.9228.
