from imgui_bundle import immapp, hello_imgui, imgui, imgui_md, ImVec2, implot
from imgui_bundle import portable_file_dialogs as pfd

import os
import cv2
import numpy as np
import pygetwindow
from vidgear.gears import CamGear

from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from parameters import *

ALL_THEMES = [
    hello_imgui.ImGuiTheme_.darcula_darker,
    hello_imgui.ImGuiTheme_.darcula,
    hello_imgui.ImGuiTheme_.imgui_colors_classic,
    hello_imgui.ImGuiTheme_.imgui_colors_dark,
    hello_imgui.ImGuiTheme_.imgui_colors_light,
    hello_imgui.ImGuiTheme_.material_flat,
    hello_imgui.ImGuiTheme_.photoshop_style,
    hello_imgui.ImGuiTheme_.gray_variations,
    hello_imgui.ImGuiTheme_.gray_variations_darker,
    hello_imgui.ImGuiTheme_.microsoft_style,
    hello_imgui.ImGuiTheme_.cherry,
    hello_imgui.ImGuiTheme_.light_rounded,
    hello_imgui.ImGuiTheme_.so_dark_accent_blue,
    hello_imgui.ImGuiTheme_.so_dark_accent_yellow,
    hello_imgui.ImGuiTheme_.so_dark_accent_red,
    hello_imgui.ImGuiTheme_.black_is_black,
    hello_imgui.ImGuiTheme_.white_is_white,
]

ALL_MODELS = [
    "DS_UCFCrimeDataset___C_Arson___DT_2023_11_22__22_40_51.h5",
    "DS_UCFCrimeDataset___C_Explosion___DT_2023_11_22__23_22_08.h5", 
    "DS_UCFCrimeDataset___C_RoadAccidents___DT_2023_11_22__23_03_08.h5",
    "DS_UCFCrimeDataset___C_Shooting___DT_2023_11_22__23_57_57.h5",
    "DS_UCFCrimeDataset___C_Vandalism___DT_2023_11_22__23_39_07.h5",
]
ALL_MODELS_NAME = ["Individual: Arson", "Individual: Explosion", "Individual: RoadAccidents",
                   "Individual: Shooting", "Individual: Vandalism"]

context = {"current_theme_index": 0, "current_model_index": 0,
           "video_type": "path", "video_path": os.getcwd(), "youtube_link": "",
           "opt_show_video": True, "cam_gear_opt": {}, "anomaly_scores_array": [],
           "normal_scores_array": [], "show_anomaly_score": True, "show_normal_score": True,
           "frame_buffer": [], "video_capture": None, "frame_count": 0, "cam_gear": None,
           "anomaly_scores": [], "model": load_model(os.path.join(os.getcwd(), "models", ALL_MODELS[0])),
           "details_header_visibility": True, "prediction_stats_header_visibility": True,
           "var_fps": f"{FPS}", "var_buffer_capacity_multiplier": "4", "variables_header_visibility": True,
           "var_buffer_capacity": f"{FRAME_COUNT}"}

def generateCamGearOptions():
    global context
    context["cam_gear_opt"] = {"CAP_PROP_FPS": FPS}

def compileModel():
    context["model"].compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

def videoPathHeader():
    global context
    nodeOpen = imgui.tree_node_ex(
        "Select video via video path",
        imgui.TreeNodeFlags_.open_on_arrow |
        imgui.TreeNodeFlags_.selected if context["video_type"] == "path" else 0)
    if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            context["video_type"] = "path"
    if nodeOpen:
        imgui.set_next_item_width(imgui.get_window_size()[0]*0.70)
        _, context["video_path"] = imgui.input_text_with_hint("##video_path", "Video path",
                                                              context["video_path"])
        imgui.same_line()
        if imgui.button("Open file", imgui.ImVec2(imgui.get_window_size()[0]*0.20, 0)):
            fileDialog = pfd.open_file("select video", os.getcwd(), ["*.mp4"])
            if len(fileDialog.result()) > 0:
                context["video_path"] = fileDialog.result()[0]
                endAnomalyDetection()
        imgui.tree_pop()

def youtubeLinkHeader():
    global context
    nodeOpen = imgui.tree_node_ex(
        "Select video via youtube link",
        imgui.TreeNodeFlags_.open_on_arrow |
        imgui.TreeNodeFlags_.selected if context["video_type"] == "youtube" else 0)
    if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
        context["video_type"] = "path"
    if nodeOpen:
        if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
            context["video_type"] = "youtube"
        imgui.set_next_item_width(imgui.get_window_size()[0]*0.72)
        _, context["youtube_link"] = imgui.input_text_with_hint("##youtube_link", "Youtube link",
                                                                context["youtube_link"])
        imgui.tree_pop()
        
def frameReduction(frames):
    skipFrameWindow = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)
    return np.array([cv2.resize(np.array(frames[i*skipFrameWindow]), IMAGE_DIMENSION) / 255
            for i in range(SEQUENCE_LENGTH)])
        
def startAnomalyDetection():
    global context
    if imgui.button("Start anomaly detection & classification", imgui.ImVec2(imgui.get_window_size()[0], 0)):
        context["frame_buffer"] = []
        context["anomaly_scores"] = [0, 0]
        context["anomaly_scores_array"] = []
        context["normal_scores_array"] = []
        if context["video_type"] == "path":
            context["video_capture"] = cv2.VideoCapture(context["video_path"])
        elif context["video_type"] == "youtube":
            generateCamGearOptions()
            context["cam_gear"] = CamGear(source=context["youtube_link"], stream_mode=True,
                                          time_delay=1, logging=True, **context["cam_gear_opt"]).start()
    if context["video_capture"] is not None or context["cam_gear"] is not None:
        if imgui.button("Restart", imgui.ImVec2(imgui.get_window_size()[0]*0.49, 0)):
            restartAnomalyDetection()
        imgui.same_line()
        if imgui.button("End", imgui.ImVec2(imgui.get_window_size()[0]*0.49, 0)):
            endAnomalyDetection()

def processOptions():
    global context
    if len(context["frame_buffer"]) > 0:
        if context["opt_show_video"]:
            cv2.imshow(context["cam_gear"].ytv_metadata["title"] if context[
                "video_type"] == "youtube" else context["video_path"], context["frame_buffer"][-1])
        else:
            cv2.destroyAllWindows()
            
def processAnomalyDetection():
    if context["video_type"] == "path" and context["video_capture"] is not None:
        if context["video_capture"].isOpened():
            ret, frame = context["video_capture"].read()
            context["frame_count"] += 1
            if ret == False:
                context["video_capture"].release()
                context["video_capture"] = None
                cv2.destroyAllWindows()
                return
            if len(context["frame_buffer"]) < FRAME_COUNT:
                context["frame_buffer"].append(frame)
                return
            frames = frameReduction(context["frame_buffer"])
            context["anomaly_scores"] = context["model"].predict(np.array([frames]))[0]
            context["anomaly_scores_array"].append(context["anomaly_scores"][0])
            context["normal_scores_array"].append(context["anomaly_scores"][1])
            context["frame_buffer"].append(frame)
            context["frame_buffer"] = context["frame_buffer"][1:]
        else:
            context["video_capture"].release()
            context["video_capture"] = None
    if context["video_type"] == "youtube" and context["cam_gear"] is not None:
        frame = context["cam_gear"].read()
        if frame is None:
            context["cam_gear"].stop()
            context["cam_gear"] = None
            cv2.destroyAllWindows()
            return
        context["frame_count"] += 1
        if len(context["frame_buffer"]) < FRAME_COUNT:
            context["frame_buffer"].append(frame)
            return
        frames = frameReduction(context["frame_buffer"])
        context["anomaly_scores"] = context["model"].predict(np.array([frames]))[0]
        context["anomaly_scores_array"].append(context["anomaly_scores"][0])
        context["normal_scores_array"].append(context["anomaly_scores"][1])
        context["frame_buffer"].append(frame)
        context["frame_buffer"] = context["frame_buffer"][1:]

def restartAnomalyDetection():
    global context
    context["frame_buffer"] = []
    context["anomaly_scores"] = [0, 0]
    context["anomaly_scores_array"] = []
    context["normal_scores_array"] = []
    context["frame_count"] = 0
    if context["video_type"] == "path":
        context["video_capture"].release()
        context["video_capture"] = cv2.VideoCapture(context["video_path"])
    elif context["video_type"] == "youtube":
        if context["cam_gear"] is not None:
            context["cam_gear"].stop()
        context["cam_gear"] = CamGear(source=context["youtube_link"], stream_mode=True,
                                      time_delay=1, logging=True).start()

def endAnomalyDetection():
    global context
    if context["video_capture"] is not None:
        context["video_capture"].release()
        context["video_capture"] = None
    if context["cam_gear"] is not None:
        context["cam_gear"].stop()
        context["cam_gear"] = None
    context["frame_buffer"] = []
    context["anomaly_scores"] = [0, 0]
    context["anomaly_scores_array"] = []
    context["normal_scores_array"] = []
    context["frame_count"] = 0
    cv2.destroyAllWindows()
    
def showStat(fieldName, fieldValue="", justLabel=False):
    imgui.table_next_column()
    imgui.text(fieldName)
    imgui.table_next_column()
    if not justLabel: imgui.text(f"{fieldValue}")

def showStatWithSlider(fieldName, fieldLabel, fieldValue):
    imgui.table_next_column()
    imgui.text(fieldName)
    imgui.table_next_column()
    imgui.set_next_item_width(imgui.get_column_width())
    imgui.slider_float(fieldLabel, fieldValue, 0, 1, flags=imgui.SliderFlags_.no_input)

def showDetails():
    global context
    context["details_header_visibility"] = imgui.collapsing_header("Details", imgui.TreeNodeFlags_.default_open)
    if not context["details_header_visibility"]:
        return
    imgui.set_next_item_width(imgui.get_window_size()[0])
    imgui.begin_table("##details_table", 2, imgui.TableFlags_.borders
                      | imgui.TableFlags_.row_bg | imgui.TableFlags_.resizable)
    showStat("Model", ALL_MODELS_NAME[context["current_model_index"]])
    if context["video_type"] == "path":
        showStat("Video path", context["video_path"])
        if context["video_capture"] is not None:
            video_fps = int(context['video_capture'].get(cv2.CAP_PROP_FPS))
            video_length = int(context['video_capture'].get(cv2.CAP_PROP_FRAME_COUNT))/\
                context['video_capture'].get(cv2.CAP_PROP_FPS)
            showStat("Video frame rate", f"{video_fps}")
            showStat("Video length", f"{video_length:.2f} secs")
    elif context["video_type"] == "youtube":
        showStat("Youtube link", context["youtube_link"])
        if context["cam_gear"] is not None:
            showStat("Youtube video title", context["cam_gear"].ytv_metadata["title"])
            showStat("Seconds elapsed", f"{(context['frame_count']/FPS):.2f}")
    imgui.end_table()
    if context["video_type"] == "path" and context["video_capture"] is not None:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        isChanged, value = imgui.slider_float("##video_timeline",
                                              context["frame_count"]/video_fps, 0, video_length)
        if isChanged:
            context["frame_count"] = int(value*video_fps)
            context["video_capture"].set(cv2.CAP_PROP_POS_FRAMES, context["frame_count"]-1)
            context["frame_buffer"] = []
            context["anomaly_scores"] = [0, 0]
            context["anomaly_scores_array"] = []
            context["normal_scores_array"] = []
    
def showPredictionStat():
    global FPS
    context["prediction_stats_header_visibility"] = imgui.collapsing_header("Prediction stats",
                                                                            imgui.TreeNodeFlags_.default_open)
    if context["prediction_stats_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        imgui.begin_table("##Prediction_stat_table", 2, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg)
        showStat("Frames elapsed", f"{context['frame_count']}")
        showStat("Current buffer size", f"{len(context['frame_buffer'])}")
        showStat("Buffer capacity", f"{FRAME_COUNT}")
        showStatWithSlider("Anomaly Probability", "##anomaly_score_slider", context['anomaly_scores'][0])
        showStatWithSlider("Normal Probability", "##normal_score_slider", context['anomaly_scores'][1])
        imgui.table_next_column()
        _, context["show_anomaly_score"] = imgui.checkbox("Show anomaly score", context["show_anomaly_score"])
        imgui.table_next_column()
        _, context["show_normal_score"] = imgui.checkbox("Show normal score", context["show_normal_score"])
        imgui.end_table()
        implot.begin_plot("scores")
        if context["show_anomaly_score"]:
            implot.plot_line("Anomaly", np.array(context["anomaly_scores_array"]), xscale=0.001,
                             flags=implot.LineFlags_.shaded)
        if context["show_normal_score"]:
            implot.plot_line("Normal", np.array(context["normal_scores_array"]), xscale=0.001,
                             flags=implot.LineFlags_.shaded)
        implot.end_plot()
        
def showVariables():
    global context
    context["variables_header_visibility"] = imgui.collapsing_header("Set variables for anomaly detection",
                                                                     imgui.TreeNodeFlags_.default_open)
    if context["variables_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        imgui.begin_table("##Variables_table", 2, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg)
        showStat("FPS",justLabel=True)
        imgui.set_next_item_width(imgui.get_column_width())
        _, context["var_fps"] = imgui.input_text("##fps_input_text", context["var_fps"],
                                                 imgui.InputTextFlags_.chars_decimal)
        showStat("Buffer capacity multiplier",justLabel=True)
        imgui.set_next_item_width(imgui.get_column_width())
        _, context["var_buffer_capacity_multiplier"] = imgui.input_text("##buffer_capacity_multiplier_input_text",
                                                                        context["var_buffer_capacity_multiplier"],
                                                                        imgui.InputTextFlags_.chars_decimal)
        showStat("Buffer capacity",justLabel=True)
        imgui.set_next_item_width(imgui.get_column_width())
        _, context["var_buffer_capacity"] = imgui.input_text("##buffer_capacity_input_text",
                                                             context["var_buffer_capacity"],
                                                             imgui.InputTextFlags_.chars_decimal)
        imgui.end_table()
        if imgui.button("Set variables", imgui.ImVec2(imgui.get_window_size()[0], 0)):
            global FPS, FRAME_COUNT
            FPS = int(context["var_fps"])
            FRAME_COUNT = int(context["var_buffer_capacity_multiplier"])*FPS
            FRAME_COUNT = int(context["var_buffer_capacity"])
            restartAnomalyDetection()

def themesMenuItem():
    global context
    for i, theme in enumerate(ALL_THEMES):
        if imgui.menu_item(theme.name, "", context["current_theme_index"] == i, True)[0]:
            context["current_theme_index"] = i
            hello_imgui.apply_theme(theme)

def modelMenuItem():
    global context
    for i, modelPath in enumerate(ALL_MODELS):
        if imgui.menu_item(ALL_MODELS_NAME[i], "", context["current_model_index"] == i, True)[0]:
            context["current_model_index"] = i
            context["model"] = load_model(os.path.join(os.getcwd(), "models", modelPath))
            compileModel()

def optionsMenuItem():
    global context
    if imgui.menu_item("Show video", "", context["opt_show_video"], True)[0]:
        context["opt_show_video"] = not context["opt_show_video"]
            
def mainMenu():
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("View", True):
            if imgui.begin_menu("Themes", True):
                themesMenuItem()
                imgui.end_menu()
            imgui.end_menu()
        if imgui.begin_menu("Model", True):
            modelMenuItem()
            imgui.end_menu()
        if imgui.begin_menu("Options", True):
            optionsMenuItem()
            imgui.end_menu()
        imgui.end_main_menu_bar()
            
def main():
    hello_imgui.apply_theme(ALL_THEMES[context["current_theme_index"]])
    mainMenu()
    imgui.dummy((0, 20))
    videoPathHeader()
    youtubeLinkHeader()
    startAnomalyDetection()
    processOptions()
    processAnomalyDetection()
    showDetails()
    if context["video_capture"] is not None or context["cam_gear"] is not None:
        showPredictionStat()
        showVariables()
    
if __name__ == "__main__":
    compileModel()
    immapp.run(
        gui_function=main,
        with_implot=True,
        window_title="Anode",
        window_size=(500, 850),
        with_markdown=True,
    )
