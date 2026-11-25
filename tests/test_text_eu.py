"""Offline tests for the European text front-ends (en de fr es it pt ru)."""
from __future__ import annotations

import time

import pytest

from pronunciada.text import get_g2p

# lang -> list of hand-checked sentences (validate + span checks run on all of them)
SENTENCES = {
    "en": [
        "I want to get a better water bottle for him.",
        "Dr. Smith's 3 cats don't like the 1990s.",
        "Most people can't hear it, but they're trying.",
        "The U.N. met on May 21st at 10:30 in New York.",
        "She paid $5.50 for a cup of coffee, 10% off.",
        "Well-known authors write books about history.",
        "Give him an apple and an orange.",
        "What a beautiful city!",
        "They've been working since 2015.",
        "\"Hello,\" said the old man - quietly.",
    ],
    "es": [
        "El niño llamó a su abuelo ayer en Bogotá.",
        "Hay 25 personas, y el 3,5% no vino.",
        "Los dedos de Gabriel están en el agua.",
        "Quiero que me digas la verdad.",
        "La ciudad de México es muy grande.",
        "¿Dónde está la estación de autobuses?",
        "Ella llegó a casa a las ocho.",
        "Me gusta el pingüino y el guiso.",
        "Juan y María comieron pan.",
        "El Sr. García vive en EE.UU. desde 1999.",
    ],
    "it": [
        "Il cane di Marco è più bello della città.",
        "Ho visto 3 amici all'università.",
        "La pratica musicale è possibile.",
        "Tutti sanno che l'Argentina vanta una squadra forte.",
        "Ci sono 1755 case nel paese.",
        "Gli scienziati hanno studiato il fenomeno.",
        "Perché non vieni a casa mia?",
        "Il gatto dorme sul divano.",
        "C'è un'altra possibilità.",
        "Le montagne più alte dell'Antartide.",
    ],
    "pt": [
        "A cidade de São Paulo tem 12 milhões de habitantes.",
        "Eles falaram com o professor ontem à tarde.",
        "O menino comeu um pão e bebeu água.",
        "Você quer ir à praia?",
        "A história do Brasil é muito interessante.",
        "Quando eles chegaram, já era noite.",
        "Os pássaros cantam na floresta.",
        "Ela trabalha numa empresa grande.",
        "Tenho 3 irmãos e uma irmã.",
        "O médico disse que está tudo bem.",
    ],
    "ru": [
        "Он сказал, что молоко хорошее.",
        "В Москве 12 миллионов жителей.",
        "Я люблю читать книги в библиотеке.",
        "Сего́дня хорошая погода.",
        "Мы были в городе вчера.",
        "Это очень интересная история.",
        "Дети играют во дворе.",
        "Солнце светит ярко.",
        "Я не знаю, где он живёт.",
        "Сколько стоит этот хлеб?",
    ],
    "de": [
        "Die CDU und es war 1984 in der Stadt.",
        "Ich habe heute keine Zeit.",
        "Der Lehrer hat den Kindern geholfen.",
        "Wir fahren morgen nach Berlin.",
        "Das Wetter ist schön, aber kalt.",
        "Sie wohnt in einem großen Haus.",
        "Kannst du mir bitte helfen?",
        "Er trinkt gern Kaffee mit Milch.",
        "Die Kinder spielen im Garten.",
        "Der Fahrradschlosswächter kam um 8 Uhr.",
    ],
    "fr": [
        "Les enfants sont allés à l'école.",
        "Il est parti avec ses amis.",
        "Nous avons 21 ans et vous êtes petits.",
        "Je ne sais pas ce qu'il veut.",
        "C'est un homme très intelligent.",
        "Les héros sont arrivés en retard.",
        "Elle a mangé une pomme rouge.",
        "Aujourd'hui, il fait beau.",
        "La SNCF a annoncé une grève.",
        "Mais où est donc Ornicar ?",
    ],
}

# (lang, sentence, word, index of the primary-stressed syllable within the word)
STRESS = [
    ("en", "The banana is delicious.", "banana", 1),
    ("en", "The banana is delicious.", "delicious", 1),
    ("en", "A photograph of a photographer.", "photographer", 1),
    ("es", "La canción es muy bonita.", "canción", 1),
    ("es", "La canción es muy bonita.", "bonita", 1),
    ("es", "El árbol y la ciudad.", "árbol", 0),
    ("es", "El árbol y la ciudad.", "ciudad", 1),
    ("es", "Los jóvenes hablan.", "jóvenes", 0),
    ("it", "La pratica musicale è possibile.", "pratica", 0),
    ("it", "La pratica musicale è possibile.", "musicale", 2),
    ("it", "La pratica musicale è possibile.", "possibile", 1),
    ("it", "La città è bella.", "città", 1),
    ("pt", "O café está quente.", "café", 1),
    ("pt", "O café está quente.", "quente", 0),
    ("pt", "A cidade é bonita.", "cidade", 1),
    ("pt", "Ele viu o animal.", "animal", 2),
    ("ru", "Он сказал, что молоко хорошее.", "молоко", 2),
    ("ru", "Он сказал, что молоко хорошее.", "сказал", 1),
    ("ru", "Город большой.", "Город", 0),
    ("ru", "Сего́дня хорошая погода.", "Сего́дня", 1),
    ("de", "Der Lehrer hat geholfen.", "Lehrer", 0),
    ("de", "Der Lehrer hat geholfen.", "geholfen", 1),
    ("de", "Die Universität ist alt.", "Universität", 4),
]

# (lang, sentence, word, canonical phone, expected alternative) - connected speech
ALTS = [
    ("en", "I want to get a better water bottle.", "better", "t", "ɾ"),        # flapping
    ("en", "I want to get a better water bottle.", "to", "u", "ə"),            # weak form
    ("en", "Give him the book.", "him", "h", ""),                             # h-dropping
    ("en", "Most people know.", "Most", "t", ""),                             # t/d dropping
    ("en", "The bottle is here.", "bottle", "ə", ""),                         # syllabic l
    ("es", "Los dedos de Gabriel.", "dedos", "d", "ð"),                       # lenition
    ("es", "Los dedos de Gabriel.", "Gabriel", "b", "β"),
    ("es", "Los dedos de Gabriel.", "Los", "s", "h"),                         # s aspiration
    ("es", "Me gusta la arena.", "la", "a", ""),                              # synalepha
    ("it", "Vado a casa.", "casa", "k", "kk"),                                # raddoppiamento
    ("it", "La casa rosa.", "casa", "s", "z"),                                # s voicing
    ("it", "Il bello.", "bello", "ɛ", "e"),                                   # e/o openness
    ("pt", "A cidade de São Paulo.", "cidade", "i", "e"),                     # final vowel
    ("pt", "A cidade de São Paulo.", "cidade", "dʒ", "d"),                    # palatalisation
    ("pt", "Ele vai falar.", "falar", "r", ""),                               # final r loss
    ("pt", "Ele come a maçã.", "come", "i", ""),                              # vowel sandhi
    ("ru", "Вот дом.", "Вот", "t", "d"),                                      # voicing across words
    ("ru", "Он сказал.", "сказал", "a", "ɑ"),                                 # measured reduction variant (round 8)
    ("de", "Wir haben es.", "haben", "ə", ""),                                # -en -> syllabic n
    ("de", "Der Lehrer.", "Lehrer", "ʀ", "ɐ"),                                # r vocalisation
    ("de", "Er ist alt.", "alt", "a", "ʔa"),                                  # glottal stop
    ("fr", "Les enfants sont arrivés.", "sont", "t", ""),                     # optional liaison
    ("fr", "Je ne sais pas.", "ne", "ə", ""),                                 # schwa elision
]

# obligatory French liaison must be in the canonical form, h aspiré blocks it
LIAISON = [
    ("Les enfants jouent.", "Les", "z", True),
    ("Les héros arrivent.", "Les", "z", False),
    ("Nous avons fini.", "Nous", "z", True),
    ("Les livres sont là.", "Les", "z", False),
]

NUMBERS = {"en": "I have 42 apples.", "es": "Tengo 42 manzanas.", "it": "Ho 42 mele.", "pt": "Tenho 42 maçãs.",
           "ru": "У меня 42 яблока.", "de": "Ich habe 42 Äpfel.", "fr": "J'ai 42 pommes."}
OOV = {"en": "The blorfingtons are here.", "es": "El blorfingón llegó.", "it": "Il blorfingone arriva.",
       "pt": "O blorfingão chegou.", "ru": "Блорфингон пришёл.", "de": "Der Blorfingtrax kommt.",
       "fr": "Le blorfingtrax arrive."}
LANGS = list(SENTENCES)


def word_phones(ref, wi):
    w = ref.words[wi]
    return ref.phones[w.phones[0]:w.phones[1]]


def find_word(ref, text):
    for i, w in enumerate(ref.words):
        if w.text == text:
            return i
    raise AssertionError(f"word {text!r} not in {[w.text for w in ref.words]}")


@pytest.mark.parametrize("lang", LANGS)
def test_sentences_validate_and_align(lang):
    g = get_g2p(lang)
    for s in SENTENCES[lang]:
        ref = g(s)
        ref.validate()
        assert ref.words, s
        prev = 0
        for w in ref.words:
            a, b = w.char_span
            assert s[a:b] == w.text, (s, w.text, w.char_span)
            assert a >= prev
            prev = b
            for si in range(*w.syllables):
                syl = ref.syllables[si]
                if lang == "fr":
                    assert syl.stress is None
                else:
                    assert syl.stress in (0, 1, 2)
        for p in ref.phones:
            for alt, prior in p.alts:
                assert prior <= 0


def test_word_counts():
    assert len(get_g2p("en")(SENTENCES["en"][0]).words) == 10
    assert [w.text for w in get_g2p("fr")("Les enfants sont allés à l'école.").words] == \
        ["Les", "enfants", "sont", "allés", "à", "l'école"]
    assert [w.text for w in get_g2p("it")("C'è un'altra possibilità.").words] == ["C'è", "un'altra", "possibilità"]
    assert [w.text for w in get_g2p("en")("Dr. Smith's 3 cats don't like the 1990s.").words] == \
        ["Dr.", "Smith's", "3", "cats", "don't", "like", "the", "1990s"]
    assert [w.text for w in get_g2p("es")("¿Dónde está?").words] == ["Dónde", "está"]


@pytest.mark.parametrize("lang,sentence,word,idx", STRESS)
def test_stress(lang, sentence, word, idx):
    ref = get_g2p(lang)(sentence)
    w = ref.words[find_word(ref, word)]
    stresses = [ref.syllables[i].stress for i in range(*w.syllables)]
    assert stresses.index(1) == idx, (word, stresses)
    assert stresses.count(1) == 1


def test_french_has_no_stress():
    ref = get_g2p("fr")("La musique est belle.")
    assert all(s.stress is None for s in ref.syllables)


def test_function_words_unstressed():
    for lang, s, w in (("en", "I went to the shop.", "the"), ("es", "Voy a la casa.", "la"),
                       ("it", "Vado in città.", "in"), ("de", "Ich bin in der Stadt.", "der"),
                       ("ru", "Он на работе.", "на"), ("pt", "Vou de carro.", "de")):
        ref = get_g2p(lang)(s)
        wi = find_word(ref, w)
        assert all(ref.syllables[i].stress == 0 for i in range(*ref.words[wi].syllables)), (lang, w)


@pytest.mark.parametrize("lang,sentence,word,phone,alt", ALTS)
def test_connected_speech_alts(lang, sentence, word, phone, alt):
    ref = get_g2p(lang)(sentence)
    ps = word_phones(ref, find_word(ref, word))
    cands = [p for p in ps if p.ipa == phone]
    assert cands, (word, [p.ipa for p in ps])
    assert any(alt in [a for a, _ in p.alts] for p in cands), (word, [(p.ipa, p.alts) for p in cands])


@pytest.mark.parametrize("sentence,word,cons,present", LIAISON)
def test_french_liaison(sentence, word, cons, present):
    ref = get_g2p("fr")(sentence)
    ps = word_phones(ref, find_word(ref, word))
    assert (ps[-1].ipa == cons) == present, [p.ipa for p in ps]


def test_french_elision_and_un():
    ref = get_g2p("fr")("L'homme a un ami.")
    assert "".join(p.ipa for p in word_phones(ref, 0)) == "lɔm"
    un = word_phones(ref, find_word(ref, "un"))
    assert any(a == "œ̃" for p in un for a, _ in p.alts)


@pytest.mark.parametrize("lang", LANGS)
def test_numbers(lang):
    ref = get_g2p(lang)(NUMBERS[lang])
    ref.validate()
    w = ref.words[find_word(ref, "42")]
    assert w.phones[1] - w.phones[0] >= 5
    assert not [x for x in ref.warnings if "42" in x]


def test_number_spellings():
    cases = {"en": ("It was 1984.", "naɪntin"), "de": ("Es war 1984.", "nɔɪntseːnhʊndəʀt"),
             "es": ("Son 3,5 kilos.", "koma"), "fr": ("Il a 80 ans.", "katʁ")}
    for lang, (s, frag) in cases.items():
        ref = get_g2p(lang)(s)
        assert frag in "".join(p.ipa for p in ref.phones), (lang, "".join(p.ipa for p in ref.phones))


@pytest.mark.parametrize("lang", LANGS)
def test_oov_fallback(lang):
    ref = get_g2p(lang)(OOV[lang])
    ref.validate()
    oov = [w for w in ref.warnings if w.startswith("oov:")]
    assert len(oov) == 1, ref.warnings
    word = oov[0][4:]
    wi = next(i for i, w in enumerate(ref.words) if w.text.lower() == word)
    assert len(word_phones(ref, wi)) >= 5
    if lang != "fr":
        w = ref.words[wi]
        assert [ref.syllables[i].stress for i in range(*w.syllables)].count(1) == 1


def test_russian_specifics():
    g = get_g2p("ru")
    ref = g("Что ты делаешь?")
    assert "".join(p.ipa for p in word_phones(ref, 0)) == "ʂto"         # irregular, from the lexicon
    ref = g("в школе и в доме")
    assert word_phones(ref, 0)[0].ipa == "f" and word_phones(ref, 3)[0].ipa == "v"   # preposition voicing
    ref = g("Ёлка")
    assert ref.syllables[0].stress == 1


def test_empty_and_symbols():
    for lang in LANGS:
        g = get_g2p(lang)
        ref = g("")
        ref.validate()
        assert ref.words == []
        ref = g("... !!! ---")
        ref.validate()


@pytest.mark.parametrize("lang", LANGS)
def test_speed(lang):
    g = get_g2p(lang)
    sents = SENTENCES[lang]
    for s in sents:
        g(s)                       # warm-up
    t0 = time.perf_counter()
    n = 0
    for _ in range(20):
        for s in sents:
            g(s + " ")             # defeat any whole-sentence caching
            n += 1
    ms = (time.perf_counter() - t0) * 1000 / n
    assert ms < 5.0, f"{lang}: {ms:.2f} ms/sentence"


@pytest.mark.parametrize("lang", LANGS)
def test_build_time(lang):
    import importlib
    mod = importlib.import_module(f"pronunciada.text.{lang}")
    t0 = time.perf_counter()
    mod.build()
    assert time.perf_counter() - t0 < 1.5
