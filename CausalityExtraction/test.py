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
        'text': "This year's Nobel Laureates in Physiology or Medicine made the remarkable and unexpected discovery that inflammation in the stomach as well as ulceration of the stomach or duodenum is the result of an infection of the stomach caused by the bacterium Helicobacter pylori."
    },
    {
        'text': 'The runtime disconnected due to inactivity.'
    },
    {
        'text': 'Deadlines might cause stress.'
    },
    {
        'text': "Bezos' space company Blue Origins was officially incorporated in Sept 2000."
    },
    {
        'text': "The damages caused by mudslides, tremors, subsidence, superficial or underground water were verified, as well as swelling clay soils."
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

print(decoded_predictions['Predictions'])