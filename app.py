
import streamlit as st
import joblib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load model and artifacts
model = joblib.load("XGBoost.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_cols = joblib.load("feature_columns.pkl")
G = joblib.load("route_graph.pkl")

label_cols = ["Ship From", "Ship To", "Commodity", "Modes"]

st.set_page_config(page_title="Supply Chain", layout="centered")
st.title("🚚 Supply Chain Cost Estimator")
st.markdown("Enter shipment details to predict distribution cost and see the best route.")

# UI Inputs
ship_from = st.selectbox("Ship From", sorted(G.nodes))
ship_to = st.selectbox("Ship To", sorted(G.nodes))
commodity = st.selectbox("Commodity", list(encoders["Commodity"].classes_))
order_value = st.number_input("Order Value (₹)", min_value=1000, step=5000, value=50000)
weight = st.number_input("Weight (KG)", min_value=1, step=50, value=1000)
volume = st.number_input("Volume (CBM)", min_value=1, step=1, value=10)

if st.button("🔍 Predict Cost and Show Route"):

    # ✅ Check for invalid input: same source and destination
    if ship_from == ship_to:
        st.warning("🚫 'Ship From' and 'Ship To' cannot be the same location.")
    else:
        try:
            all_routes = []
            for path in nx.all_simple_paths(G, source=ship_from, target=ship_to, cutoff=4):
                total_cost = 0
                total_time = 0
                modes = []
                for i in range(len(path) - 1):
                    edge = G[path[i]][path[i+1]]
                    total_cost += edge["cost"]
                    total_time += edge["time"]
                    modes.append(edge["mode"])

                mode_combo = " → ".join(modes)

                # Handle unseen mode combos silently
                if mode_combo not in encoders["Modes"].classes_:
                    updated_classes = list(encoders["Modes"].classes_) + [mode_combo]
                    encoders["Modes"].classes_ = np.array(sorted(set(updated_classes)))

                modes_encoded = encoders["Modes"].transform([mode_combo])[0]

                all_routes.append({
                    "path": path,
                    "route_str": " → ".join(path),
                    "modes": mode_combo,
                    "transit_time": total_time,
                    "modes_encoded": modes_encoded
                })

            if not all_routes:
                st.warning("🚫 No valid route found between selected locations.")
            else:
                best = min(all_routes, key=lambda r: r["transit_time"])

                input_data = {
                    "Ship From": encoders["Ship From"].transform([ship_from])[0],
                    "Ship To": encoders["Ship To"].transform([ship_to])[0],
                    "Commodity": encoders["Commodity"].transform([commodity])[0],
                    "Order Value": order_value,
                    "Weight (KG)": weight,
                    "Volume": volume,
                    "Modes": best["modes_encoded"],
                    "Total Transit Time (hrs)": best["transit_time"]
                }

                df_input = pd.DataFrame([input_data])[feature_cols]
                predicted_cost = model.predict(df_input)[0]

                # Show result
                st.success(f"💰 Predicted Distribution Cost: ₹{round(predicted_cost, 2)}")
                st.info(f"📦 Best Route: {best['route_str']}")
                st.write(f"🚛 Travel Modes: {best['modes']}")
                st.write(f"⏱️ Total Transit Time: {best['transit_time']} hours")

                # Draw graph
                st.subheader("📍 Route Visualization")
                fig, ax = plt.subplots(figsize=(12, 8))
                pos = nx.spring_layout(G, seed=42, k=0.5)  # Better spacing

                # Draw nodes and edges
                nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", ax=ax)
                nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", ax=ax)

                # Edge labels
                edge_labels = nx.get_edge_attributes(G, "mode")
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

                # Node labels with background
                for node, (x, y) in pos.items():
                    ax.text(x, y + 0.05, node, fontsize=9, ha='center',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

                # Highlight best route
                path_edges = list(zip(best['path'], best['path'][1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color="red", ax=ax)
                nx.draw_networkx_nodes(G, pos, nodelist=best['path'], node_color="orange", node_size=900, ax=ax)

                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
