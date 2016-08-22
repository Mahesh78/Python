## A Wordgame: Hangman

# Part 1: Is The Word Guessed?

def isWordGuessed(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''
    
    for s in secretWord:
        if s not in lettersGuessed:
            return False
    return True
    
# Part 2: Printing Out The User's Guess

def getGuessedWord(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''
    # FILL IN YOUR CODE HERE...
    ans = ''
    for s in secretWord:
        if s in lettersGuessed:
            ans += s
        else:
            ans += ' _ '
    return ans
    
# Part 3: Printing Out All Available Letters

def getAvailableLetters(lettersGuessed):
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    # FILL IN YOUR CODE HERE...
    import string
    Remaining = ''
    for s in string.ascii_lowercase:
        if s not in lettersGuessed:
            Remaining += s
    return Remaining

# Part 4: Hangman Part 2: The Game

def hangman(secretWord):

    # FILL IN YOUR CODE HERE...
    print 'Welcome to the game, Hangman!'
    print 'I am thinking of a word that is ' + str(len(secretWord)) + ' letters long.\n'
    print '-------------'
    guesses = 8
    wrongGuess = 0
    available = string.ascii_lowercase
    guessed = ''
    
    while wrongGuess < 8 and not isWordGuessed(secretWord, guessed):
        print 'you have ' + str(guesses - wrongGuess)+ ' guesses left'
        print 'Available letters: ' + available
        letters = raw_input('Please guess a letter: ')
        letters = letters.lower()
        if letters in guessed:
            print("Oops! You've already guessed that letter: " +
            getGuessedWord(secretWord, guessed))
        
        elif letters in secretWord:
            guessed += letters
            print 'Good guess: ' + getGuessedWord(secretWord, guessed)
        
        else:
            wrongGuess += 1
            guessed += letters
            print('Oops! That letter is not in my word: '+ 
                getGuessedWord(secretWord, guessed))
        available = getAvailableLetters(guessed)
        print '------------'
        
    if isWordGuessed(secretWord, guessed):
        print 'Congratulations, you won!'
    else:
        print 'Sorry, you ran out of guesses. The word was ' + str(secretWord) + '.'