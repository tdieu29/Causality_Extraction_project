from CausalityExtraction import input_processing, predict


texts = [
    {
        'text': 'Standing in the sun makes me sweat.'
    },
    {
        'text': 'Smoking tobacco is by far the leading cause of lung cancer.'
    },
    {
        'text': 'Consuming too mush sugar is a contributing factor to getting diabetes.'
    },
    {
        'text': 'In 2004, Elon Musk and Jeff Bezos met for a meal to discuss space.'
    },
    {
        'text': 'It was one of their few in-person interactions.'
    },
    {
        'text': "The conversation they had perfectly captures the different approaches they've taken to space exploration."
    },
    {
        'text': 'Deadlines might cause stress.'
    },
    {
        'text': "Bezos' space company Blue Origins was officially incorporated in Sept 2000."
    },
    {
        'text': "The path to success isn't straight."
    },
    {
        'text': 'Arguments lead to their friendship fallout.'
    },
    {
        'text': 'Talent starts with self-belief.'
    },
    {
        'text': 'Growth is the art of consistently starting when you are not ready.'
    },
    {
        'text': 'Strong winds lead to power outages.'
    },
    {
        'text': 'Due to strong winds, trees fell on homes.'
    },
    {
        'text': 'I arrived late because of the traffic jam.'
    },
    {
        'text': 'Traffic is a function of flow, density of the vehicles and the speed of the vehicles.'
    }
]


texts = [item['text'] for item in texts]

input_processing.get_input(texts)

decoded_predictions = predict.predict()

print(decoded_predictions)