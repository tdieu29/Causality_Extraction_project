from CausalityExtraction import input_processing, predict
import streamlit as st
import re  
import pandas as pd 

st.title('Causality Extraction from texts')

text = st.text_area(
label='Please enter up to 16 sentences. Below are several examples: ', 
value=
"01. Standing in the sun makes me sweat. \n02. Smoking tobacco is by far the leading cause of lung cancer. \n03. Consuming \
too mush sugar is a contributing factor to getting diabetes. \n04. In 2004, Elon Musk and Jeff Bezos met for a meal to discuss \
space. \n05. It was one of their few in-person interactions. \n06. The runtime disconnected due to inactivity. \n07. Deadlines \
might cause stress. \n08. Bezos' space company Blue Origins was officially incorporated in Sept 2000. \n09. The path to \
success isn't straight. \n10. Arguments lead to their friendship fallout. \n11. Talent starts with self-belief. \n12. Growth \
is the art of consistently starting when you are not ready. \n13. Strong winds lead to power outages. \n14. Due to strong \
winds, trees fell on homes. \n15. I arrived late because of the traffic jam. \n16. Traffic is a function of flow, density of \
the vehicles and the speed of the vehicles.", 
height=430)
            
sentences = []


if st.button('Get results'):
    input_sentences = re.findall('\d{2}\.(.+)', text)
    for input_sentence in input_sentences:
        sentence = input_sentence.strip()
        sentences.append(sentence)
    
    with st.spinner('Running...'):   
        input_processing.get_input(sentences)
        decoded_predictions = predict.predict()
    st.success('Done!')

    result = {'Sentence': [i for i in range(1,17)],
                'Cause': [], 
                'Effect': []}

    for i in range(0,16):
        if len(decoded_predictions['Predictions'][i]) != 0:
            cause = decoded_predictions['Predictions'][i][0][0]
            effect = decoded_predictions['Predictions'][i][0][1]
            result['Cause'].append(cause)
            result['Effect'].append(effect)
        else:
            cause = 'Non-causal sentence'
            effect = 'Non-causal sentence'
            result['Cause'].append(cause)
            result['Effect'].append(effect)
    
    df = pd.DataFrame(data=result)

    df = df.to_markdown(index=False)
    st.markdown(df)

    
