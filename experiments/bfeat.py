import re, unicodedata, numpy as np, pandas as pd
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
SENT = re.compile(r"[.!?。！？\n]")
NAMES = ['char_count','word_count','sentence_count','line_count','mean_word_length','mean_sentence_words','word_length_std','unique_word_ratio',
 'uppercase_ratio','digit_ratio','whitespace_ratio','non_ascii_ratio','punctuation_per_100_chars',
 'blank_line_ratio','heading_per_100_lines','bullet_per_100_lines','numbered_per_100_lines','code_fence_per_100_lines',
 'bold_per_100_chars','link_per_100_chars'] + [p+'_per_100_chars' for p in ['period','comma','question','exclamation','colon','semicolon','hyphen','em_dash','parenthesis','quote','asterisk','backtick']]
PCH = {'period':'.','comma':',','question':'?','exclamation':'!','colon':':','semicolon':';','hyphen':'-','em_dash':'—','parenthesis':'()','quote':'"\'“”‘’','asterisk':'*','backtick':'`'}
def bfeat(t):
    n = len(t)
    if n == 0: return [0.0]*32
    words = WORD.findall(t); wl = [len(w) for w in words]; nw = len(words)
    sents = [s for s in SENT.split(t) if s.strip()]
    lines = t.split('\n'); L = len(lines)
    f = [n, nw, len(sents), L, np.mean(wl) if wl else 0, nw/len(sents) if sents else 0, np.std(wl) if wl else 0,
         len(set(w.lower() for w in words))/nw if nw else 0,
         sum(c.isupper() for c in t)/n, sum(c.isdigit() for c in t)/n, sum(c.isspace() for c in t)/n, sum(ord(c)>127 for c in t)/n,
         sum(unicodedata.category(c).startswith('P') for c in t)/n*100,
         sum(not l.strip() for l in lines)/L,
         sum(bool(re.match(r'\s*#{1,6}\s', l)) for l in lines)/L*100,
         sum(bool(re.match(r'\s*[-*+•]\s', l)) for l in lines)/L*100,
         sum(bool(re.match(r'\s*\d+[.)]\s', l)) for l in lines)/L*100,
         sum(l.strip().startswith('```') for l in lines)/L*100,
         len(re.findall(r'\*\*[^*]+\*\*', t))/n*100, len(re.findall(r'\[[^\]]+\]\([^)]+\)', t))/n*100]
    # quote: ASCII " and ' are counted? B: "ASCII 및 곡선형 따옴표 합산" -> use " “ ” ‘ ’ and '
    f += [sum(t.count(c) for c in PCH[k])/n*100 for k in ['period','comma','question','exclamation','colon','semicolon','hyphen','em_dash','parenthesis','quote','asterisk','backtick']]
    return f
def B32(texts): return pd.DataFrame([bfeat(t) for t in texts], columns=NAMES)
