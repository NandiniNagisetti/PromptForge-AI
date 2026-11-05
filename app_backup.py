
import streamlit as st
import pandas as pd
import requests
import re
import matplotlib.pyplot as plt

from prompt_variants import generate_prompt_variants

st.set_page_config(
    page_title="Prompt Sensitivity Lab",
    page_icon="🧪",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

BENCHMARK = [
    {
        "question": "What is the capital of France?",
        "expected": "Paris",
        "category": "Geography"
    },
    {
        "question": "What is 12 multiplied by 8?",
        "expected": "96",
        "category": "Mathematics"
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "expected": "Mars",
        "category": "Science"
    },
    {
        "question": "What gas do plants absorb from the atmosphere?",
        "expected": "carbon dioxide",
        "category": "Science"
    },
    {
        "question": "How many days are there in a standard week?",
        "expected": "7",
        "category": "General Knowledge"
    }
]


def ask_model(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42
            }
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_correct(response, expected):
    answer = normalize(response)
    target = normalize(expected)

    if not answer or not target:
        return False

    # Basic reference matching for this factual benchmark.
    return target in answer


st.title("🧪 Prompt Sensitivity Lab")
st.caption("Language Model Evaluation With Prompt Sensitivity Quantified")

st.write(
    "Evaluate how prompt wording affects model correctness "
    "across multiple benchmark questions."
)

st.sidebar.header("Experiment Configuration")
st.sidebar.write(f"Model: `{MODEL}`")

number_of_questions = st.sidebar.selectbox(
    "Benchmark size",
    [3, 5],
    index=0
)

st.sidebar.info(
    "Each question is tested using five prompt variants. "
    "Start with three questions to reduce runtime."
)

st.header("1. Benchmark Dataset")

dataset = BENCHMARK[:number_of_questions]

st.dataframe(
    pd.DataFrame(dataset),
    use_container_width=True,
    hide_index=True
)

if st.button("Run Complete Benchmark", type="primary"):
    results = []
    total = len(dataset) * 5
    completed = 0

    progress = st.progress(0)
    status = st.empty()

    try:
        for case_number, case in enumerate(dataset, start=1):
            variants = generate_prompt_variants(case["question"])

            for variant_name, variant_prompt in variants.items():
                status.write(
                    f"Question {case_number}/{len(dataset)} — "
                    f"Evaluating: {variant_name}"
                )

                answer = ask_model(variant_prompt)

                results.append({
                    "Question": case["question"],
                    "Category": case["category"],
                    "Variant": variant_name,
                    "Prompt": variant_prompt,
                    "Expected Answer": case["expected"],
                    "Model Response": answer,
                    "Correct": is_correct(
                        answer,
                        case["expected"]
                    ),
                    "Score": int(
                        is_correct(answer, case["expected"])
                    )
                })

                completed += 1
                progress.progress(completed / total)

        st.session_state["benchmark_results"] = pd.DataFrame(results)
        status.success("Benchmark completed successfully!")

    except requests.exceptions.ConnectionError:
        status.error(
            "Cannot connect to Ollama. Ensure Ollama is running "
            "and the model is installed."
        )

    except requests.exceptions.Timeout:
        status.error(
            "The model request timed out. Retry with three questions "
            "or check that the model is responding."
        )

    except requests.exceptions.RequestException as error:
        status.error(f"Model request failed: {error}")


if "benchmark_results" in st.session_state:
    df = st.session_state["benchmark_results"]

    st.header("2. Evaluation Results")

    st.dataframe(
        df[
            [
                "Question",
                "Variant",
                "Model Response",
                "Expected Answer",
                "Correct"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.header("3. Aggregate Metrics")

    overall_accuracy = df["Score"].mean()

    # Calculate the average absolute accuracy difference
    # between each variation and the original prompt,
    # across the same benchmark questions.
    pivot = df.pivot(
        index="Question",
        columns="Variant",
        values="Score"
    )

    variation_columns = [
        col for col in pivot.columns if col != "Original"
    ]

    if "Original" in pivot.columns and variation_columns:
        differences = pivot[variation_columns].sub(
            pivot["Original"],
            axis=0
        ).abs()

        sensitivity = differences.to_numpy().mean()
    else:
        sensitivity = float("nan")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Answer-Match Rate",
        f"{overall_accuracy:.1%}"
    )

    col2.metric(
        "Questions Evaluated",
        df["Question"].nunique()
    )

    col3.metric(
        "Mean Absolute Score Change",
        f"{sensitivity:.3f}"
        if pd.notna(sensitivity) else "N/A"
    )

    st.caption(
        "Sensitivity is the mean absolute change in binary answer-match "
        "scores between the original prompt and its variations. "
        "It ranges from 0 to 1. It measures changes in correctness, "
        "not all aspects of model behavior."
    )

    st.header("4. Accuracy by Prompt Variant")

    variant_accuracy = (
        df.groupby("Variant", sort=False)["Score"]
        .mean()
        .rename("Accuracy")
    )

    st.bar_chart(variant_accuracy)

    st.header("5. Accuracy by Category")

    category_accuracy = (
        df.groupby(["Category", "Variant"])["Score"]
        .mean()
        .unstack()
    )

    st.dataframe(category_accuracy, use_container_width=True)

    st.header("6. Export Results")

    st.download_button(
        "Download Detailed Results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prompt_sensitivity_benchmark.csv",
        mime="text/csv"
    )

    st.success(
        "Experiment results are ready to review and export."
    )

    st.info(
        "Research limitation: the built-in benchmark is small, and "
        "substring-based answer matching can produce false positives. "
        "Validate answers and use task-appropriate metrics before "
        "reporting research conclusions."
    )
