from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def run_evaluation(
    questions: list,
    ground_truths: list,
    answers: list,
    contexts: list,
    llm,
    embeddings_model
) -> dict:
    """
    Run RAGAS evaluation on a set of questions and answers.
    Returns dict of metric scores.
    """
    try:
        dataset = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })

        ragas_llm  = LangchainLLMWrapper(llm)
        ragas_emb  = LangchainEmbeddingsWrapper(embeddings_model)

        results = evaluate(
            dataset    = dataset,
            metrics    = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ],
            llm        = ragas_llm,
            embeddings = ragas_emb,
        )

        df = results.to_pandas()

        def safe_mean(col):
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    v = float(vals.mean())
                    return round(v, 3)
            return None

        return {
            "faithfulness":      safe_mean("faithfulness"),
            "answer_relevancy":  safe_mean("answer_relevancy"),
            "context_precision": safe_mean("context_precision"),
            "context_recall":    safe_mean("context_recall"),
            "dataframe":         df,
            "error":             None
        }

    except Exception as e:
        return {
            "faithfulness": None, "answer_relevancy": None,
            "context_precision": None, "context_recall": None,
            "dataframe": None, "error": str(e)
        }