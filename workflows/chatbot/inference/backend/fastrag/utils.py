
def detect_language(query):
    is_english = all(ord(c) < 128 for c in query)
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

    if is_english and not is_chinese:
        return 'English'
    elif is_chinese and not is_english:
        return 'Chinese'
    else:
        return 'Mixed'
