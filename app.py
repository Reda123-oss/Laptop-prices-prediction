from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and data
model = joblib.load("best_xgboost_model.pkl")
df = pd.read_csv("laptop_prices_filtered.csv")

# Load encoders (must be saved previously)
company_enc = joblib.load("company_encoder.pkl")
# Vérification du contenu de l'encodeur après son chargement
print("Classes de l'encodeur : ", company_enc.classes_)

product_enc = joblib.load("product_encoder.pkl")
os_enc = joblib.load("OS_encoder.pkl")
cpu_enc = joblib.load("CPU_company_encoder.pkl")
cpu_model_enc = joblib.load("CPU_model_encoder.pkl")
storage_enc = joblib.load("PrimaryStorageType_encoder.pkl")
gpu_enc = joblib.load("GPU_company_encoder.pkl")
gpu_model_enc = joblib.load("GPU_model_encoder.pkl")
retina_enc = joblib.load("RetinaDisplay_encoder.pkl")
touch_enc = joblib.load("Touchscreen_encoder.pkl")

# Add others as needed...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mode1', methods=['POST'])
def mode1():
    try:
        # Récupérer les champs depuis le formulaire
        company = request.form.get("Company")
        product = request.form.get("Product")
        os = request.form.get("OS")
        cpu_company = request.form.get("CPU_company")
        cpu_model = request.form.get("CPU_model")
        cpu_freq = float(request.form.get("CPU_freq"))
        gpu_company = request.form.get("GPU_company")
        gpu_model = request.form.get("GPU_model")
        ram = int(request.form.get("Ram"))
        PrimaryStorage = int(request.form.get("PrimaryStorage"))
        storage_type = request.form.get("PrimaryStorageType")
        inches = float(request.form.get("Inches"))
        retina = request.form.get("RetinaDisplay")
        touchscreen = request.form.get("Touchscreen")
        ScreenH = int(request.form.get("ScreenH"))
        ScreenW = int(request.form.get("ScreenW"))

        # company = request.form["Company"]
        # product = request.form["Product"]
        # os = request.form["OperatingSystem"]
        # cpu_company = request.form["CPU_company"]
        # cpu_model = request.form["CPU_model"]
        # cpu_freq = float(request.form["CPU_freq"])
        # gpu_company = request.form["GPU_company"]
        # gpu_model = request.form["GPU_model"]
        # ram = int(request.form["Ram"])
        # PrimaryStorage = int(request.form["PrimaryStorage"])
        # storage_type = request.form["PrimaryStorageType"]
        # inches = float(request.form["Inches"])
        # retina = request.form["RetinaDisplay"]
        # touchscreen = request.form["Touchscreen"]
        # ScreenH = request.form["ScreenH"]
        # ScreenW = request.form["ScreenW"]

        test_company = "Apple"
        test_product = "MacBook Pro"
        test_inches = 13.3
        test_ram = 8
        test_PrimaryStorage = 128
        test_ScreenW = 2560
        test_ScreenH = 1600
        test_touchscreen = "No"
        test_retina = "Yes"
        test_os = "macOS"
        test_cpu_freq = 2.3
        test_cpu_company = "Intel"
        test_cpu_model = "Core i5"
        test_storage_type = "SSD"
        test_gpu_company = "Intel"
        test_gpu_model = "Iris Plus Graphics 640"

        # --- Comparaison des valeurs ---
        print("=== Comparaison Formulaire vs Test ===")
        print(f"Company: {company} == {test_company} -> {company == test_company}")

        print(f"Product: {product} == {test_product} -> {product == test_product}")
        print(f"Inches: {inches} == {test_inches} -> {inches == test_inches}")
        print(f"Ram: {ram} == {test_ram} -> {ram == test_ram}")
        print(f"PrimaryStorage: {PrimaryStorage} == {test_PrimaryStorage} -> {PrimaryStorage == test_PrimaryStorage}")
        print(f"ScreenW: {ScreenW} == {test_ScreenW} -> {ScreenW == test_ScreenW}")
        print(f"ScreenH: {ScreenH} == {test_ScreenH} -> {ScreenH == test_ScreenH}")
        print(f"Touchscreen: {touchscreen} == {test_touchscreen} -> {touchscreen == test_touchscreen}")
        print(f"RetinaDisplay: {retina} == {test_retina} -> {retina == test_retina}")
        print(f"OperatingSystem: {os} == {test_os} -> {os == test_os}")
        print(f"CPU_freq: {cpu_freq} == {test_cpu_freq} -> {cpu_freq == test_cpu_freq}")
        print(f"CPU_company: {cpu_company} == {test_cpu_company} -> {cpu_company == test_cpu_company}")
        print(f"CPU_model: {cpu_model} == {test_cpu_model} -> {cpu_model == test_cpu_model}")
        print(f"PrimaryStorageType: {storage_type} == {test_storage_type} -> {storage_type == test_storage_type}")
        print(f"GPU_company: {gpu_company} == {test_gpu_company} -> {gpu_company == test_gpu_company}")
        print(f"GPU_model: {gpu_model} == {test_gpu_model} -> {gpu_model == test_gpu_model}")

        # print("==== Données reçues du formulaire ====")
        # print(f"Company           : {company}")
        # print(f"Product           : {product}")
        # print(f"Operating System  : {os}")
        # print(f"CPU Company       : {cpu_company}")
        # print(f"CPU Model         : {cpu_model}")
        # print(f"CPU Frequency     : {cpu_freq} GHz")
        # print(f"GPU Company       : {gpu_company}")
        # print(f"GPU Model         : {gpu_model}")
        # print(f"RAM               : {ram} GB")
        # print(f"Storage           : {PrimaryStorage} GB")
        # print(f"Storage Type      : {storage_type}")
        # print(f"Inches            : {inches}\"")
        # print(f"Retina Display    : {retina}")
        # print(f"Touchscreen       : {touchscreen}")
        # print(f"Screen Height     : {ScreenH} px")
        # print(f"Screen Width      : {ScreenW} px")
        # print("======================================")


       
        # Encodage des colonnes catégorielles
        company_encoded = company_enc.transform([company])[0]
        product_encoded = product_enc.transform([product])[0]
        os_encoded = os_enc.transform([os])[0]
        cpu_company_encoded = cpu_enc.transform([cpu_company])[0]
        cpu_model_encoded = cpu_model_enc.transform([cpu_model])[0]
        gpu_company_encoded = gpu_enc.transform([gpu_company])[0]
        gpu_model_encoded = gpu_model_enc.transform([gpu_model])[0]
        storage_encoded = storage_enc.transform([storage_type])[0]
        retina_encoded = retina_enc.transform([retina])[0]
        touchscreen_encoded = touch_enc.transform([touchscreen])[0]

        print("==== Valeurs encodées ====")
        print("Company encodé          :", company_encoded)
        print("Product encodé          :", product_encoded)
        print("OS encodé               :", os_encoded)
        print("CPU Company encodé      :", cpu_company_encoded)
        print("CPU Model encodé        :", cpu_model_encoded)
        print("GPU Company encodé      :", gpu_company_encoded)
        print("GPU Model encodé        :", gpu_model_encoded)
        print("Storage Type encodé     :", storage_encoded)
        print("Retina Display encodé   :", retina_encoded)
        print("Touchscreen encodé      :", touchscreen_encoded)
        print("==========================")

        # Construction du DataFrame pour la prédiction
        input_df = pd.DataFrame([{
            "Company": company_encoded,
            "Product": product_encoded,
            "Inches": inches,
            "Ram": ram,
            "OS": os_encoded,
            "ScreenW": ScreenW,
            "ScreenH": ScreenH,
            "Touchscreen": touchscreen_encoded,
            "RetinaDisplay": retina_encoded,
            "CPU_company": cpu_company_encoded,
            "CPU_freq": cpu_freq,
            "CPU_model": cpu_model_encoded,
            "PrimaryStorage": PrimaryStorage,
            "PrimaryStorageType": storage_encoded,
            "GPU_company": gpu_company_encoded,
            "GPU_model": gpu_model_encoded
        }])

        print("==== Données encodées prêtes pour la prédiction ====")
        print(input_df)

        display_features = {
                "Company": company,
                "Product": product,
                "Inches": inches,
                "RAM (Go)": ram,
                "OS": os,
                "Screen Width (px)": ScreenW,
                "Screen Height (px)": ScreenH,
                "Touchscreen": touchscreen,
                "Retina Display": retina,
                "CPU Company": cpu_company,
                "CPU Frequency (GHz)": cpu_freq,
                "CPU Model": cpu_model,
                "Primary Storage (Go)": PrimaryStorage,
                "Primary Storage Type": storage_type,
                "GPU Company": gpu_company,
                "GPU Model": gpu_model
            }

        predicted_price = model.predict(input_df)[0]
        print(predicted_price)
        
        mae = 164.04

        mse = 71907.77

        rscore = 0.8747      # Remplace avec ta vraie valeur

        # Ensuite tu envoies à ton template prediction.html 
       
        recommended = df[(df['Price_euros'] >= predicted_price - 100) &
                         (df['Price_euros'] <= predicted_price + 100)]
        return render_template(
                    'prediction.html',
                    predicted_price=predicted_price,
                    features=display_features,
                    mae=mae,
                    mse=mse,
                    rscore=rscore
                )
        # return render_template('prediction.html', features=display_features, predicted_price=predicted_price)

    except ValueError as e:
        return f"Erreur : {str(e)}"
    except Exception as e:
        return f"Une erreur est survenue : {str(e)}"

@app.route('/mode2', methods=['POST'])
def mode2():
    min_price = float(request.form['MinPrice'])
    max_price = float(request.form['MaxPrice'])

    filtered = df[(df['Price_euros'] >= min_price) & (df['Price_euros'] <= max_price)]

    return render_template('results.html', results=filtered.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
