# FastAPI model schemas

from typing import List

from fastapi import Query
from pydantic import BaseModel, validator

class Text(BaseModel):
    text: str = Query(None, min_length=1)

class PredictPayload(BaseModel):
    texts: List[Text]

    @validator('texts')
    def list_not_empty(cls, value):
        if not len(value):
            raise ValueError('List of texts cannot be empty.')
        if len(value) % 16 != 0:
            raise ValueError('Length of list must be a multiple of 16.')
        return value
    
    class Config:
        schema_extra = {
            'example': {
                'texts': [
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
            }
        }
