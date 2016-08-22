## Google RSS feed

# Part 1: Data Structure Design

class NewsStory(object):
    def __init__(self,guid, title, subject, summary, link):
        self.g = guid
        self.t = title
        self.s = subject
        self.summ = summary
        self.l = link
    def getGuid(self):
        return self.g
    def getTitle(self):
        return self.t
    def getSubject(self):
        return self.s
    def getSummary(self):
        return self.summ
    def getLink(self):
        return self.l
        
# Part 2: Word Triggers

class WordTrigger(Trigger):
    def __init__(self,word):
        self.word = word
    def isWordIn(self,text):
        t = ''
        for char in text:
            t += char.strip(string.punctuation).lower()
        t = t.split()
        for word in t:
            if word == self.word.lower():
                return True
            return False
class TitleTrigger(WordTrigger):
    def evaluate(self,story):
        return self.isWordIn(story.getTitle)

class SubjectTrigger(WordTrigger):
    def evaluate(self,story):
        return self.isWordIn(story.getSubject)
        
class SummaryTrigger(WordTrigger):
    def evaluate(self,story):
        return self.isWordIn(story.getSummary)