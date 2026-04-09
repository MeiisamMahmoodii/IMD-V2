from isomorphic.banned_words_extractor import BannedWordsExtractor

def test_banned_word_extract():
    ext = BannedWordsExtractor()
    words = ext.extract("Building strong protective barrier to exclude unsafe actors from community")
    assert len(words) >= 5
