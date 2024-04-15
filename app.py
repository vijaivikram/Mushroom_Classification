from flask import Flask,render_template,request
from src.MushroomClassification.pipelines.prediction_pipeline import CustomData,PredictPipeline 
from src.MushroomClassification.logger import logging

app = Flask(__name__)
app.static_folder="static"
app.template_folder="static/templates"


@app.route("/")
def Mushroom():
    return render_template("home.html")


@app.route("/predict",methods=["GET","POST"])
def Predict():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(
            cap_shape=request.form.get("cap-shape"),cap_color=request.form.get("cap-color"),
            cap_surface=request.form.get("cap-surface"),bruises=request.form.get("bruises"),
            odor=request.form.get("odor"),gill_attachment=request.form.get("gill-attachment"),
            gill_spacing=request.form.get("gill-spacing"),gill_size=request.form.get("gill-size"),
            gill_color=request.form.get("gill-color"),stalk_shape=request.form.get("stalk-shape"),
            stalk_root=request.form.get("stalk-root"),stalk_surface_above_ring=request.form.get("stalk-surface-above-ring"),
            stalk_surface_below_ring=request.form.get("stalk-surface-below-ring"),
            stalk_color_above_ring=request.form.get("stalk-color-above-ring"),
            stalk_color_below_ring=request.form.get("stalk-color-below-ring"),
            veil_type = request.form.get('veil-type'),
            veil_color=request.form.get("veil-color"),
            ring_number=request.form.get("ring-number"),ring_type=request.form.get("ring-type"),
            spore_print_color=request.form.get("spore-print-color"),population=request.form.get("population"),
            habitat=request.form.get("habitat")
            )

        features=data.get_data_as_dataframe()
        logging.info(f'feature data{features.to_string()}')
        pred=PredictPipeline().predict(features=features)
        if pred=="e":
            pred="This Mushroom is Edible"
        else:
            pred="This Mushroom is Poisonous!"

        return render_template("home.html",final_result=pred)        
    



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)