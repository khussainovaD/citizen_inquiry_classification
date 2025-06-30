import pytest
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
from string import punctuation

# Настройка Natasha
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

def preprocess_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    tokens = []
    for token in doc.tokens:
        if token.text.lower() not in punctuation:
            token.lemmatize(morph_vocab)
            tokens.append(token.lemma.lower())
    return " ".join(tokens)

@pytest.fixture
def sample_text():
    return "Здравствуйте! Я хочу пожаловаться на задержку выплаты пособий."

def test_output_is_string(sample_text):
    result = preprocess_text(sample_text)
    assert isinstance(result, str), "Output must be a string"

def test_expected_lemmas_present(sample_text):
    result = preprocess_text(sample_text)
    result_tokens = result.split()
    expected_lemmas = {"здравствовать", "хотеть", "пожаловаться", "задержка", "выплата", "пособие"}
    for lemma in expected_lemmas:
        assert lemma in result_tokens, f"Lemma '{lemma}' not found in result"

def test_punctuation_removed(sample_text):
    result = preprocess_text(sample_text)
    for punct in punctuation:
        assert punct not in result, f"Punctuation '{punct}' should be removed"
