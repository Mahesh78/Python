## Encryption & Decryption:

# Part 1: Encryption

def buildCoder(shift):
    """
    Returns a dict that can apply a Caesar cipher to a letter.
    The cipher is defined by the shift value. Ignores non-letter characters
    like punctuation, numbers, and spaces.
    """
    ### TODO 
    dic = {}
    l = string.lowercase
    m = string.uppercase
    r = m + l
    
    for i in range(len(l)):
        dic[m[i]] = m[(i+shift)%26]
        dic[l[i]] = l[(i+shift)%26]
        
    return dic
    
def applyCoder(text, coder):
    """
    Applies the coder to the text. Returns the encoded text.
    """
    txt = ''
    for char in text:
        if char in string.punctuation or char == ' ' or char in string.digits:
            txt += char
        else:
            txt += coder[char]
    return txt

def applyShift(text, shift):
    """
    Given a text, returns a new text Caesar shifted by the given shift
    offset. Lower case letters should remain lower case, upper case
    letters should remain upper case, and all other punctuation should
    stay as it is.
    """
    return applyCoder(text, buildCoder(shift))
    
# Part 2: Decryption

def findBestShift(wordList, text):
    """
    Finds a shift key that can decrypt the encoded text.

    """

    realWords = 0
    bestShift = 0
    #wordsText = text.split(' ')
    for shift in range(26):
        num = 0
        deWord = applyShift(text, shift)
        wordsText = deWord.split(' ')
        for wor in range(len(wordsText)):
            if isWord(wordList,wordsText[wor]):
                 num += 1
        if num > realWords:
            realWords = num
            bestShift = shift
    return bestShift
    
def decryptStory():
    """
    Using the methods you created in this problem set,
    decrypt the story given by the function getStoryString().
    
    returns: string - story in plain text
    """

    storyEncrypt = getStoryString()
    wordList = loadWords()
    return applyShift(storyEncrypt, findBestShift(wordList,storyEncrypt))