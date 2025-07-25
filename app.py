from typing import get_type_hints

import pandas as pd
from pydantic import TypeAdapter
import streamlit as st

from house_price_prediction.modeling.predict import load_best_model, load_preprocessed_data
from house_price_prediction.modeling.schemas import HouseFeatures


@st.cache_data
def load_preprocessed_model():
    return load_preprocessed_data()


@st.cache_data
def load_trained_model():
    return load_best_model()


def main():
    preprocessor = load_preprocessed_model()
    model = load_trained_model()

    # Initialize session state
    if "filters" not in st.session_state:
        st.session_state.filters = [{"field": None, "value": ""}]

    st.title("🏠 Ames House Price Predictor")
    st.markdown("Enter the features below and get a prediction of the sale price.")

    # Get model field names and types
    fields = get_type_hints(HouseFeatures)
    field_names = list(fields.keys())

    # Draw current filters
    for i, filter_row in enumerate(st.session_state.filters):
        cols = st.columns([3, 4, 1])
        selected_field = cols[0].selectbox(
            f"Field #{i + 1}",
            field_names,
            key=f"field_{i}",
            index=field_names.index(filter_row["field"]) if filter_row["field"] else 0,
        )
        value = cols[1].text_input(f"Value #{i + 1}", value=filter_row["value"], key=f"value_{i}")
        if cols[2].button("❌", key=f"remove_{i}"):
            st.session_state.filters.pop(i)
            st.rerun()
        else:
            st.session_state.filters[i]["field"] = selected_field
            st.session_state.filters[i]["value"] = value

    # Add new filter row
    if st.button("➕ Add filter", "add_filter_button"):
        st.session_state.filters.append({"field": None, "value": ""})
        st.rerun()

    # Predict
    if st.button("🚀 Predict", "predict_button"):
        input_data = {}
        for f in st.session_state.filters:
            if f["field"] and f["value"] != "":
                # Cast to correct type
                target_type = fields[f["field"]]
                try:
                    input_data[f["field"]] = TypeAdapter(target_type).validate_strings(f["value"])
                except ValueError or TypeError:
                    st.error(f"Invalid value for {f['field']}")
                    st.stop()

        full_input_data = HouseFeatures(**input_data)
        df = pd.DataFrame([full_input_data.model_dump()])
        X_processed = preprocessor.transform(df) if preprocessor else df
        prediction = model.predict(X_processed)[0]
        st.success(f"💰 Estimated Sale Price: ${prediction:,.0f}")


main()
