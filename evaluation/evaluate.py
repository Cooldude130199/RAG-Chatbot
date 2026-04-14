from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
import pandas as pd

def run_eval(queries, answers, contexts):
    dataset = {
        "question": queries,
        "answer": answers,
        "contexts": contexts
    }
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print(df)
    return df
