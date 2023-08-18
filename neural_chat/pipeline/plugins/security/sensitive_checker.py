"""Function to check the intent of the input user query with LLM."""
from __future__ import division
import os

def Q2B(query):
    """Converting Full-width Characters to Half-width Characters."""
    content = ""
    for uchar in query:
        mid_char = ord(uchar)
        if mid_char == 12288:
            mid_char = 32
        elif (mid_char > 65280 and mid_char < 65375):
            mid_char -= 65248
        content += chr(mid_char)
    return content


class SensitiveChecker:
    def __init__(self, dict_path=None, matchType=2):
        if dict_path == None or (not os.path.exists(dict_path)):
            dict_path = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(dict_path, "stopword.txt"), encoding="utf8")
        self.Stopwords = [i.split('\n')[0] for i in f.readlines()]
        f1 = open(os.path.join(dict_path, "dict.txt"), encoding="utf8")
        lst = f1.readlines()
        self.sensitiveWordSet = [i.split("\n")[0].split("\t") for i in lst]
        self.sensitiveWordMap = self.initSensitiveWordMap()
        self.matchType = matchType
        
    
    def initSensitiveWordMap(self):
        """Initialize the sensitive word dictory."""
        sensitiveWordTree = dict()
        for category, key in self.sensitiveWordSet:
            if type(key) == 'unicode' and type(category) == 'unicode':
                pass
            else:
                key = str(key)
                category = str(category)

            nowNode = sensitiveWordTree
            word_count = len(key)
            for i in range(word_count):
                subChar = key[i]
                wordNode = nowNode.get(subChar)
                if wordNode != None:
                    nowNode = wordNode
                else:
                    newNode = dict()
                    newNode["isEnd"] = False
                    nowNode[subChar] = newNode
                    nowNode = newNode
                if i == word_count - 1:
                    nowNode["isEnd"] = True
                    nowNode["category"] = category
        return sensitiveWordTree

    def contains(self, txt):
        """Check if the input text contain the sensitive words."""
        flag = False
        for i in range(len(txt)):
            matchFlag = self.checkSensitiveWord(txt, i)[0]
            if matchFlag > 0:
                flag = True
        return flag


    def checkSensitiveWord(self, txt, beginIndex):
        """Check if the input token contains sensitive word."""
        flag = False
        category = ""
        matchFlag = 0
        nowMap = self.sensitiveWordMap
        tmpFlag = 0
        for i in range(beginIndex, len(txt)):
            word = txt[i]
            if word in self.Stopwords and len(nowMap) < 100:
                tmpFlag += 1
                continue
            nowMap = nowMap.get(word)
            if nowMap:
                matchFlag += 1
                tmpFlag += 1
                if nowMap.get("isEnd") == True:
                    flag = True
                    category = nowMap.get("category")
                    if self.matchType == 1:
                        break
            else:
                break
        if matchFlag < 2 or not flag:
            tmpFlag = 0
        return tmpFlag, category
    
    
    def sensitive_check(self, context):
        txt_convert = Q2B(context)
        contain = self.contains(txt=txt_convert)
        return contain
    
    
    def get_sensitive_word(self, context):
        """get the sensitive word."""
        sensitiveWordList = list()
        for i in range(len(context)):
            length = self.checkSensitiveWord(context, i)[0]
            category = self.checkSensitiveWord(context, i)[1]
            if length > 0:
                word = context[i:i + length]
                sensitiveWordList.append(category + ":" + word)
                i = i + length - 1
        return sensitiveWordList

    
    def sensitive_filter(self, context, replaceChar="*"):
        """Replace the sensitive word."""
        tupleSet = self.get_sensitive_word(context)
        wordSet = [i.split(":")[1] for i in tupleSet]
        resultTxt = ""
        if len(wordSet) > 0:
            for word in wordSet:
                replaceString = len(word) * replaceChar
                context = context.replace(word, replaceString)
                resultTxt = context
        else:
            resultTxt = context
        return resultTxt
        
        
        
        
    
