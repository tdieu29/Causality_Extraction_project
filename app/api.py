# FastAPI application endpoints

from datetime import datetime
from functools import wraps
from http import HTTPStatus
from fastapi import FastAPI, Request
import uvicorn

from app.schemas import PredictPayload
from CausalityExtraction import input_processing, predict


# Define application
app = FastAPI()


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""
    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            'message': results['message'],
            'method': request.method,
            'status_code': results['status_code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url._url
        }

        # Add data
        if 'data' in results:
            response['data'] = results['data'] 
        return response
    return wrap


@app.get('/')
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'data': {}
    }
    return response


@app.post('/predict')
@construct_response
def _best_predict(request: Request, payload: PredictPayload):
    """Predictions for a list of texts using the best model. """
    # Predict
    texts = [item.text for item in payload.texts]
    input_processing.get_input(texts)
    decoded_predictions = predict.predict()

    response = {
        'message': HTTPStatus.OK.phrase,
        'status_code': HTTPStatus.OK,
        'data': {
            'Predictions': decoded_predictions['Predictions']
        }
    }
    return response

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
#uvicorn app.api:app --host 127.0.0.1 --port 8001 --reload