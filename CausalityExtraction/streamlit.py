from CausalityExtraction import input_processing, predict
import streamlit as st
import re  
import pandas as pd 

# Title of webpage
st.title('Causality Extraction from texts')

# Input box on webpage
text = st.text_area(
label='The box below contains 16 example sentences. Please replace them with your own sentences and enter up to 16 sentences. \
If you did not enter as many as 16 sentences, please leave the remaining example sentences in the box.', 
value=
"01. Standing in the sun makes me sweat. \n02. Smoking tobacco is by far the leading cause of lung cancer. \n03. Consuming too mush \
sugar is a contributing factor to getting diabetes. \n04. In 2004, Elon Musk and Jeff Bezos met for a meal to discuss space. \n05. \
This year's Nobel Laureates in Physiology or Medicine made the remarkable and unexpected discovery that inflammation in the stomach \
as well as ulceration of the stomach or duodenum is the result of an infection of the stomach caused by the bacterium Helicobacter \
pylori. \n06. The runtime disconnected due to inactivity. \n07. Deadlines might cause stress. \n08. Bezos' space company Blue Origins \
was officially incorporated in Sept 2000. \n09. The damages caused by mudslides, tremors, subsidence, superficial or underground water \
were verified, as well as swelling clay soils. \n10. Arguments lead to their friendship fallout. \n11. Talent starts with self-belief.\
 \n12. Growth is the art of consistently starting when you are not ready. \n13. Strong winds lead to power outages. \n14. Due to strong \
winds, trees fell on homes. \n15. I arrived late because of the traffic jam. \n16. Traffic is a function of flow, density of the vehicles \
and the speed of the vehicles.", 
height=430)

# When user clicks on the 'Get results' button:
sentences = []

if st.button('Get results'):
    input_sentences = re.findall('\d{2}\.(.+)', text) # Extract individual sentences from the input text
    for input_sentence in input_sentences:
        sentence = input_sentence.strip() # Strip white spaces at the beginning and end of each sentence 
        sentences.append(sentence)
    
    with st.spinner('Running...'):   
        inputProcessor = input_processing
        inputProcessor.get_input(sentences)
        #input_processing.get_input(sentences)
        decoded_predictions = predict.predict()
    st.success('Done!')

    result = {'Sentence': [],
                'Cause - Effect pairs': [], }

    for i in range(16):
        prediction = decoded_predictions['Predictions'][i]
        if len(prediction) != 0:
            for m in range(len(prediction)):
                result['Sentence'].append(i+1)
                result['Cause - Effect pairs'].append(prediction[m])
        else:
            result['Sentence'].append(i+1)
            result['Cause - Effect pairs'].append('None')

    
    df = pd.DataFrame(data=result)
    df = df.to_markdown(index=False)
    st.markdown(df)