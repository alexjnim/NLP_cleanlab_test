import nltk
import unicodedata
from utils.contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import spacy
import collections
import en_core_web_sm
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import wordnet

tokenizer = ToktokTokenizer()
nlp = en_core_web_sm.load()


def strip_html_tags(text):
    """
    Removes html tags
    Input
    ----------
    text (string): Text or a url.
    Returns
    ----------
    The stripped text
    """
    cleanr = re.compile("<.*?>")
    stripped_text = re.sub(cleanr, " ", text)
    return stripped_text


def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):

    text = nlp(text)
    text = " ".join(
        [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
    )
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = (
            contraction_mapping.get(match)
            if contraction_mapping.get(match)
            else contraction_mapping.get(match.lower())
        )
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_accented_chars(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r"[^a-zA-Z0-9\s]|\[|\]" if not remove_digits else r"[^a-zA-Z\s]|\[|\]"
    text = re.sub(pattern, "", text)
    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.remove("no")
    stopwords.remove("not")

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def remove_repeated_characters(text):
    repeat_pattern = re.compile(r"(\w*)(\w)\2(\w*)")
    match_substitution = r"\1\2\3"

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in text]
    return correct_tokens


def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_stemming=False,
    text_lemmatization=True,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    lower_case=True,
):

    normalized_corpus = []
    count = 0
    # normalize each document in the corpus
    for doc in corpus:
        count = count + 1
        if count % 100 == 0:
            print("Normalizing text: {}".format(count))

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # stem text
        if text_stemming and not text_lemmatization:
            doc = simple_porter_stemming(doc)

        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r"([{.(-)!}])")
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)

        # remove extra whitespace
        doc = re.sub(" +", " ", doc)

        # lower the cases all of words
        if lower_case:
            doc = doc.lower()

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc)

        # remove extra whitespace
        doc = re.sub(" +", " ", doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
