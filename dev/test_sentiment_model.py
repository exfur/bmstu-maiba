from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
)

# Testing English and French
print(classifier("I love how fast this model runs on my laptop!"))
print(classifier("C'est une perte de temps complète."))
print(classifier("Я ненавижу всех"))

classifier = pipeline(
    "text-classification", model="tabularisai/multilingual-sentiment-analysis"
)

# Testing Russian and Spanish
print(classifier("Этот проект превзошел все мои ожидания!"))
print(classifier("El servicio fue bastante mediocre, la verdad."))
print(classifier("Я ненавижу всех"))
