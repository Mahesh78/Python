## A Word Game: Words With Friends / Scrabble

# Part 1: Word Scores

def getWordScore(word, n):
    """
    Returns the score for a word. Assumes the word is a valid word.
    """
    score = 0
    SUM = 0
    for char in word:
        SUM += SCRABBLE_LETTER_VALUES[char]
    if len(word) == n:
        score = (SUM)*len(word) + 50
    else:
        score = (SUM)*len(word)
    
    return score
    
# Part 2: Dealing With Hands

def updateHand(hand, word):
    """
    Assumes that 'hand' has all the letters in word.
    In other words, this assumes that however many times
    a letter appears in 'word', 'hand' has at least as
    many of that letter in it.
    
    returns: dictionary (string -> int)
    """
    updated = hand.copy()
    for letter in word:
        updated[letter] = updated[letter] - 1
    return updated

# Part 3: Valid Words

def isValidWord(word, hand, wordList):
    """
    Returns True if word is in the wordList and is entirely
    composed of letters in the hand. Otherwise, returns False.

    """

    if word not in wordList:
        return False
    for i in word:
        if word.count(i) > hand.get(i, 0):
            return False
    return True
    
    
# Part 4: Hand Length

def calculateHandlen(hand):
    """ 
    Returns the length (number of letters) in the current hand.
    
    hand: dictionary (string int)
    returns: integer
    """
    h = 0
    for i in range(len(hand)):
        h += hand[hand.keys()[i]]
    return h

# Part 5: Playing a Hand

def playHand(hand, wordList, n):
    """
    Allows the user to play the given hand.
    """

    score = 0
    total = hand.copy()
    while calculateHandlen(total) > 0:
        print 'Current Hand: ',
        print (displayHand(total))
        word =  raw_input('Enter word, or a "." to indicate that you are finished: ')
        if word == '.':
            break
        else:
            if isValidWord(word, total, wordList) == False:
                print ('Invalid word, please try again.\n')
            else:
                score += getWordScore(word, n)
                print '"'+str(word) + '" earned '+str(getWordScore(word, n))+' points. Total: '+str(score)+' points.'
                
                total = updateHand(total, word)

   
    if calculateHandlen(total) > 0:
        print 'Goodbye! Total score: '+str(score)+ ' points.'
    else:
        print 'Run out of letters. Total score: '+ str(score)  + ' points.'


# Part 6: Playing a Game

def playGame(wordList):
    """
    Allow the user to play an arbitrary number of hands.
 
    """

    userInput = None
    newHand = None
    while userInput != 'e':
        userInput = raw_input('Enter n to deal a new hand, '
        'r to replay the last hand, or e to end game: ')
        
        if userInput == 'n':
            newHand = dealHand(HAND_SIZE)
            playHand(newHand.copy(), wordList, HAND_SIZE)
            print 
        elif userInput == 'r':
            if newHand == None:
                print ('You have not played a hand yet.'
                ' Please play a new hand first!')
                print 
                
            else:
                playHand(newHand.copy(), wordList, HAND_SIZE)
                    
        
        elif userInput != 'e':    
                        print 'Invalid command'


# Part 7: Computer Chooses a Word

def compChooseWord(hand, wordList, n):
    """
    Given a hand and a wordList, find the word that gives 
    the maximum value score, and return it.

    """
    maxScore = 0

    bestWord = None
    for i in wordList:
        if isValidWord(i,hand,wordList):
            score = getWordScore(i,n)
            if score > maxScore:
                maxScore = score
                bestWord = i
    return bestWord
    
# Part 8: Computer Plays a Hand

def compChooseWord(hand, wordList, n):
    """
    Given a hand and a wordList, find the word that gives 
    the maximum value score, and return it.
    """
    maxScore = 0

    bestWord = None
    for i in wordList:
        if isValidWord(i,hand,wordList):
            score = getWordScore(i,n)
            if score > maxScore:
                maxScore = score
                bestWord = i
    # return the best word you found.
    return bestWord

def compPlayHand(hand, wordList, n):
    """
    Allows the computer to play the given hand, following the same procedure
    as playHand, except instead of the user choosing a word, the computer 
    chooses it.

    """
    def displayNewHand(hand):
        disp = ''
        for letter in hand.keys():
            for j in range(hand[letter]):
                disp += letter + ' '
        return disp
    newHand = hand.copy()
    newScore = 0
    while compChooseWord(newHand, wordList, n):
        print "Current Hand:  " + displayNewHand(newHand)
        newWord = compChooseWord(newHand, wordList, n)
        scores = getWordScore(newWord,n)
        newScore += scores
        print '"'+ newWord + '" ' + "earned " + str(scores) + " points. Total: " + str(newScore) + " points"
        print
        
        newHand = updateHand(newHand, newWord)
    
    if sum(newHand.values()) != 0:
        print "Current Hand:  " + displayNewHand(newHand)
    print "Total score: " + str(newScore) + " points"
    
# Part 9: You and Your Computer.

def playGame(wordList):
    """
    Allow the user to play an arbitrary number of hands.

    wordList: list (string)
    """
    newHands = None
    userIn = raw_input('Enter n to deal a new hand, '
        'r to replay the last hand, or e to end game: ')
    while userIn != 'e':
        
        if userIn == 'n':
            newHands = dealHand(HAND_SIZE)
        elif userIn == "r":
            if newHands == None:
                print ('You have not played a hand yet.'
                ' Please play a new hand first!')
                userIn = raw_input("Enter n to deal a new hand, r to replay the " + 
                    "last hand, or e to end game: ")
                print
                continue
        else:
            print 'Invalid command.'
            print
            userIn = raw_input("Enter n to deal a new hand, r to replay the " + 
                "last hand, or e to end game: ")
            print
            continue

        playerTurn = raw_input("Enter u to have yourself play, c to have the " + 
            "computer play: ")

        while playerTurn != "c" and playerTurn != "u":
            print 'Invalid command'
            print
            
            playerTurn = raw_input("Enter u to have yourself play, c to have " + 
                "the computer play: ")

        if playerTurn == "u":
            playHand(newHands, wordList, HAND_SIZE)
        
        else:
            compPlayHand(newHands, wordList, HAND_SIZE)
            

        userIn = raw_input("Enter n to deal a new hand, r to replay the last " + 
            "hand, or e to end game: ")
        print