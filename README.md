# Algorithm Explorer

An interactive Streamlit app for exploring the search algorithms in `searching.py`.
The dependencies include Streamlit, TensorFlow, and Flask; the current app uses
Streamlit, while TensorFlow and Flask are available for future features.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Edit the graph in the app as JSON. Each node maps to its neighbors and edge costs,
for example:

```json
{
  "A": {"B": 2, "C": 4},
  "B": {"A": 2},
  "C": {"A": 4}
}
```

For A* searches, provide a finite, non-negative heuristic for every node.
Results show traversal order, not a reconstructed route.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Create an app in Streamlit Community Cloud and select that repository.
3. Set the main file path to `app.py` and deploy.

Community Cloud installs the packages listed in `requirements.txt`.
