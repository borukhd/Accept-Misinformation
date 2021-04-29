from empath import Empath


class SentimentAnalyzer:
    #Interface for Empath module

    responses = []
    count = {}
    count2 = {}
    initialized = False
    lexicon = Empath()
    relevant = ['negative_emotion', 'health', 'dispute', 'government', 'leisure', 'healing', 'military', 'fight', 'meeting', 'shape_and_size', 'power', 'terrorism', 'competing', 'optimism', 'sexual', 'zest', 'love', 'joy', 'lust', 'office', 'money', 'aggression', 'wealthy', 'banking', 'kill', 'business', 'fabric', 'speaking', 'work', 'valuable', 'economics', 'clothing', 'payment', 'feminine', 'worship', 'affection', 'friends', 'positive_emotion', 'giving', 'help', 'school', 'college', 'real_estate', 'reading', 'gain', 'science', 'negotiate', 'law', 'crime', 'stealing', 'white_collar_job', 'weapon', 'night', 'strength']
    result = {}
    result2 = {}

    #codes and headlines dictionaries for experiment 1 (text) and experiment 2 and 3 (text2)
    text = {
            'Real1': "Border Patrol Chief on migrant crisis: We're seeing increase after increase,' no signs of slowing",
            'Real2': "Conservative groups form coalition to 'agressively' oppose socialized medicine in US",
            'Real3': "Georgia Representative: Democrats are 'blind with rage' in their desire to impeach President Trump",
            'Real4': "Centrists warn of 'slippery slope' ater Democrats skirt rules to fund agenda",
            'Real5': "'He's Making Things Happen.' Trump Fans Rally in Washington, D.C.",
            'Real6': "McDaniel: 'I love my uncle' Romney, but GOP should unite behind Trump",
            'Real7': "Rep. Seth Mould Slams Joe Biden's Pro-Iraq War Vote \n The 2020 Democratic candidate's critique came after Biden publicly changed...",
            'Real8': "Trump to donate $1 million to Texas recovery \n President Trump will donate $1 million of his fortune to recovery...",
            'Real9': "Trump vows proposed tax cuts will benefit middle class \n President Trump on Wednesday said his proposed tax cuts will benefit the...",
            'Real10': "Anger in France, Britain over Trump's gun law speech \n U. S. President Donald Trump caused Anger in France and Britain by...",
            'Real11': "At Trump's big-city hotels, business dropped as his political star rose, internal documents show",
            'Real12': "Investment Boom From Trump's Tax Cut Has Yet to Appear \n Analysts are still waiting for hard evidence that the new tax law is setting o...",
            'Real13': "Ex-White House ethics chief: Trump's Mar-a-Lago is a 'symbol of corruption'",
            'Real14': "U.S. Stocks Tumble After Trump Announces New Import Tariffs \n The Dow Jones Industrial Average tumbled more than 400 points, erasing...",
            'Real15': "Trump admits his cabinet had 'some clinkers' \n Adapted from 'The Best People: Trump's Cabinet and the Siege on...",
            'Real16': "Trump Exaggerates Mueller Team's Ties to Obama and Democrats",
            'Real17': "Trump Supporter Arrested After Allegedly Threatening To Kill Members of Congress",
            'Real18': "Summer Zervos, Trump Accuser, Subpoenas 'The Apprentice' Recordings",
            'Fake1': "CORONER'S REPORT: Woman Found On Clinton Estate Was Dead 15 Years, Suffered Torture And Malnutrition",
            'Fake2': "BIG Democrat Just Hailed Out Of Disney World In Handcuffs Screaming 'I Can Do Whatever I Want!'",
            'Fake3': "BREAKING: Over 500 'Migrant Caravaners' Arrested With Suicide Vests",
            'Fake4': "Denzel Washington: With Trump We Avoided War With Russia And Orwellian Police State",
            'Fake5': "Kenya: Authorities Release Barack Obama's \"Real\" Birth Certificate",
            'Fake6': "Maine House Democrats Vote To Allow Female Genital Mutilation",
            'Fake7': "Nancy Peloi's Son Arrested For Murder",
            'Fake8': "Trump Reveal Which Democratic President Was Also KKK Member, Liberals In Meltdown Mode",
            'Fake9': "Trump's Top Republican Rival Surrenders, Endorses Donald For 2020",
            'Fake10': "Ahead Of His Possible Arrest, Jared Kushner Secretly Leaves The Country",
            'Fake11': "Hispanic Woman Claims, \"Donald Trump Paid Me For Sex In Cancun, This Is Our Love Child\"",
            'Fake12': "Michelle Obama Says She Plans To Run Against Trump in 2020 - With Barack As Her V.P.!",
            'Fake13': "Trump Pays Rudy Giuliane $130,000 To Stay Silent From Now On",
            'Fake14': "Trump Threatens To Cancel Visit To Israel Because \"The McDonald's There Doesn't Have A Bacon Cheese Big Mac\"",
            'Fake15': "President Trump Readies Deportation of Melania After Huge Fight At White House",
            'Fake16': "Trump Wants To Deport American Indians To India",
            'Fake17': "Trump's Top Scientist Pick: \"Scientists Are Just Dumb Regular People That Think Dinosaurs Existed And The Earth Is Getting Warmer\"",
            'Fake18': "W.H. Staffers Defect, Releasing Private Tape Recording That Has Trump Silent",
        }
    @staticmethod
    def initialize():
        if SentimentAnalyzer.initialized:
            return
        for a in SentimentAnalyzer.text.keys():
            SentimentAnalyzer.result[a] = SentimentAnalyzer.lexicon.analyze(SentimentAnalyzer.text[a], normalize=False)
        SentimentAnalyzer.filterFrequency()
        SentimentAnalyzer.initialized = True


    @staticmethod
    def filterFrequency():  
        #drops sentiments that occur less than "minimumFreq" times in any headline; including every sentiment would lead to exponential overhead in model optimization functions.  
        sentiments = []
        for task in SentimentAnalyzer.result.keys():
            sentiments.extend([a for a in SentimentAnalyzer.result[task].keys()])
        for task in SentimentAnalyzer.text.keys():
            for a in sentiments:
                if a not in SentimentAnalyzer.count.keys():
                    SentimentAnalyzer.count[a] = 0
                SentimentAnalyzer.count[a] += float(SentimentAnalyzer.result[task][a]) if a in SentimentAnalyzer.result[task].keys() else 0
        sortedFreq = {k: v for k, v in sorted(SentimentAnalyzer.count.items(), key=lambda item: item[1])}
        #minimumFreq = 7
        #SentimentAnalyzer.relevant = [a for a in SentimentAnalyzer.count.keys() if float(SentimentAnalyzer.count[a]) > minimumFreq]
        SentimentAnalyzer.relevant = [a for a in sortedFreq.keys()][-10:]
        print(SentimentAnalyzer.relevant)

    @staticmethod
    def filterFrequency_onlyRelevant():  
        #drops sentiments that occur less than "minimumFreq" times in any headline; including every sentiment would lead to exponential overhead in model optimization functions.  
        for a in SentimentAnalyzer.relevant:
            SentimentAnalyzer.count[a] = 0 
        for task in SentimentAnalyzer.text.keys():
            for a in SentimentAnalyzer.relevant:
                SentimentAnalyzer.count[a] += float(SentimentAnalyzer.result[task][a])
                SentimentAnalyzer.count[a] = SentimentAnalyzer.count[a] 
        x = SentimentAnalyzer.count
        minimumFreq = 6
        SentimentAnalyzer.relevant = [a for a in SentimentAnalyzer.count.keys() if float(SentimentAnalyzer.count[a]) > minimumFreq]
        print(SentimentAnalyzer.relevant)

    @staticmethod
    def an_dict(task):
        #returns sentiment value for an item's headline as calcuted by Empath.
        if not SentimentAnalyzer.initialized:
            SentimentAnalyzer.initialize()
        return SentimentAnalyzer.result[task]
    @staticmethod
    def analysis(item):
        #returns sentiment value for an item's headline as calcuted by Empath.
        if not SentimentAnalyzer.initialized:
            SentimentAnalyzer.initialize()
        return SentimentAnalyzer.result[item.task[0][0]]