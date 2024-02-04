from imgui_bundle import immapp, hello_imgui, imgui, imgui_md, ImVec2
from imgui_bundle import portable_file_dialogs as pfd

import os
import cv2
import numpy as np
import pygetwindow

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
ALL_THEMES_NAME = [theme.name for theme in ALL_THEMES]

ALL_MODELS = [
    "DS_UCFCrimeDataset___C_Arson___DT_2023_11_22__22_40_51.h5",
    "DS_UCFCrimeDataset___C_Explosion___DT_2023_11_22__23_22_08.h5", 
    "DS_UCFCrimeDataset___C_RoadAccidents___DT_2023_11_22__23_03_08.h5",
    "DS_UCFCrimeDataset___C_Shooting___DT_2023_11_22__23_57_57.h5",
    "DS_UCFCrimeDataset___C_Vandalism___DT_2023_11_22__23_39_07.h5",
    "",
]
ALL_MODELS_NAME = ["Individual: Arson", "Individual: Explosion", "Individual: RoadAccidents",
                   "Individual: Shooting", "Individual: Vandalism"]

context = {"select_theme_header_visibility": False, "current_theme_index": 6,
           "select_window_header_visibility": True, "windows": [], "current_window": 0,
           "current_model_visibility": True, "current_model_index": 0,
           "select_video_path_header_visibility": True, "current_video_path": os.getcwd(),
           "frame_buffer": [], "video_capture": None, "frame_count": 0, "anomaly_scores": [],
           "model": load_model(os.path.join(os.getcwd(), "models", ALL_MODELS[0])),
           "video_options_header_visibility": True, "prediction_stats_header_visibility": True,
           "var_fps": f"{FPS}", "var_buffer_size_multiplier": "7", "variables_header_visibility": True}

def compileModel():
    context["model"].compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

def currentTheme():
    global context
    context["select_theme_header_visibility"] = imgui.collapsing_header("Select application theme")
    if context["select_theme_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        isChanged, context["current_theme_index"] = imgui.list_box("##Select_theme_list_box",
                                                                   context["current_theme_index"],
                                                                   ALL_THEMES_NAME)
        if isChanged:
            hello_imgui.apply_theme(ALL_THEMES[context["current_theme_index"]])

def currentWindow():
    global context
    context["select_window_header_visibility"] = imgui.collapsing_header("Select video from window",)
    if context["select_window_header_visibility"]:
        context["windows"] = [title for title in pygetwindow.getAllTitles() if len(title) != 0]
        imgui.set_next_item_width(imgui.get_window_size()[0])
        _, context["current_window"] = imgui.list_box("##Select_window_to_capture_list_box",
                                                      context["current_window"], context["windows"])
        
def currentModel():
    global context, ALL_MODELS
    context["select_model_header_visibility"] = imgui.collapsing_header("Select model for anomaly detection",
                                                                         imgui.TreeNodeFlags_.default_open)
    if context["select_model_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        isChanged, context["current_model_index"] = imgui.list_box(
            "##Select_model_for_anomaly_detection_list_box", context["current_model_index"], ALL_MODELS_NAME)
        if isChanged:
            context["model"] = load_model(os.path.join(os.getcwd(), "models",
                                                       ALL_MODELS[context["current_model_index"]]))
            compileModel()
            endAnomalyDetection()
        imgui.set_next_item_width(imgui.get_window_size()[0]*0.8)    
        imgui.input_text("##Model_path_input_text",
                         os.path.join(os.getcwd(), "models", ALL_MODELS[context["current_model_index"]]),
                         imgui.InputTextFlags_.read_only)
        imgui.same_line()
        if imgui.button("Select model", imgui.ImVec2(imgui.get_window_size()[0]*0.18, 0)):
            fileDialog = pfd.open_file("select model", os.getcwd(), ["*.h5"])
            if len(fileDialog.result()) > 0:
                ALL_MODELS[context["current_model_index"]] = fileDialog.result()[0]
                context["model"] = load_model(fileDialog.result()[0])
                compileModel()
                endAnomalyDetection()

def currentVideoPath():
    global context
    context["select_video_path_header_visibility"] = imgui.collapsing_header("Select video from path",
                                                                             imgui.TreeNodeFlags_.default_open)
    if context["select_video_path_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0]*0.8)    
        _, context["current_video_path"] = imgui.input_text_with_hint("##Video_path_input_text",
                                                                      "video path",
                                                                      context["current_video_path"])
        imgui.same_line()
        if imgui.button("Open file", imgui.ImVec2(imgui.get_window_size()[0]*0.18, 0)):
            fileDialog = pfd.open_file("select video", os.getcwd(), ["*.mp4"])
            if len(fileDialog.result()) > 0:
                context["current_video_path"] = fileDialog.result()[0]
                endAnomalyDetection()

def frameReduction(frames):
    skipFrameWindow = max(int(FRAME_COUNT / SEQUENCE_LENGTH), 1)
    return np.array([cv2.resize(np.array(frames[i*skipFrameWindow]), IMAGE_DIMENSION) / 255
            for i in range(SEQUENCE_LENGTH)])
        
def startAnomalyDetection():
    global context
    if imgui.button("Start anomaly detection & classification", imgui.ImVec2(imgui.get_window_size()[0], 0)):
        context["frame_buffer"] = []
        context["video_capture"] = cv2.VideoCapture(context["current_video_path"])
        context["anomaly_scores"] = [0, 0]
        
def processAnomalyDetection():
    global context
    if context["video_capture"] != None:
        if context["video_capture"].isOpened():
            ret, frame = context["video_capture"].read()
            context["frame_count"] += 1
            if ret == False:
                context["video_capture"].release()
                context["video_capture"] = None
            if len(context["frame_buffer"]) < FRAME_COUNT:
                context["frame_buffer"].append(frame)
                return
            frames = frameReduction(context["frame_buffer"])
            context["anomaly_scores"] = context["model"].predict(np.array([frames]))[0]
            context["frame_buffer"].append(frame)
            context["frame_buffer"] = context["frame_buffer"][1:]
        else:
            context["video_capture"].release()
            context["video_capture"] = None

def restartAnomalyDetection():
    global context
    context["frame_buffer"] = []
    context["video_capture"].release()
    context["video_capture"] = None
    context["video_capture"] = cv2.VideoCapture(context["current_video_path"])
    context["anomaly_scores"] = [0, 0]
    context["frame_count"] = 0

def endAnomalyDetection():
    global context
    if context["video_capture"] is not None:
        context["video_capture"].release()
        context["video_capture"] = None
    context["frame_buffer"] = []
    context["anomaly_scores"] = [0, 0]
    context["frame_count"] = 0
    
def ExitAnomalyDetection():
    global context
    imgui.table_next_column()
    if imgui.button("Restart", imgui.ImVec2(imgui.get_column_width(), 0)):
        restartAnomalyDetection()
    imgui.table_next_column()
    if imgui.button("End", imgui.ImVec2(imgui.get_column_width(), 0)):
        endAnomalyDetection()
        
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

def showVideoOptions():
    global context
    context["video_options_header_visibility"] = imgui.collapsing_header("Video Options",
                                                                         imgui.TreeNodeFlags_.default_open)
    if context["video_options_header_visibility"]:
        video_fps = int(context['video_capture'].get(cv2.CAP_PROP_FPS))
        video_length = int(context['video_capture'].get(cv2.CAP_PROP_FRAME_COUNT))/context[
            'video_capture'].get(cv2.CAP_PROP_FPS)
        imgui.set_next_item_width(imgui.get_window_size()[0])
        imgui.begin_table("##Video_options_table", 2, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg)
        showStat("Video frame rate", f"{video_fps}")
        showStat("Video length", f"{video_length:.2f} secs")
        imgui.end_table()
        imgui.set_next_item_width(imgui.get_window_size()[0])
        isChanged, value = imgui.slider_float("##video_timeline",
                                              context['frame_count']/video_fps, 0, video_length)
        if isChanged:
            context['frame_count'] = value*video_fps
            context['video_capture'].set(cv2.CAP_PROP_POS_FRAMES, context['frame_count']-1)
            context['frame_buffer'] = []
            context['anomaly_scores'] = [0, 0]
    
def showPredictionStat():
    global FPS
    context["prediction_stats_header_visibility"] = imgui.collapsing_header("Prediction stats",
                                                                            imgui.TreeNodeFlags_.default_open)
    if context["prediction_stats_header_visibility"]:
        imgui.set_next_item_width(imgui.get_window_size()[0])
        imgui.begin_table("##Prediction_stat_table", 2, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg)
        showStat("Frame count", f"{context['frame_count']}")
        showStat("Buffer size (buffer_size_multiplier * FPS)", f"{FRAME_COUNT}")
        showStat("Seconds passed", f"{context['frame_count']/FRAME_COUNT:.2f} secs")
        showStatWithSlider("Anomaly Probability", "##anomaly_score_slider", context['anomaly_scores'][0])
        showStatWithSlider("Normal Probability", "##normal_score_slider", context['anomaly_scores'][1])
        ExitAnomalyDetection()
        imgui.end_table()
        
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
        showStat("Buffer size multiplier",justLabel=True)
        imgui.set_next_item_width(imgui.get_column_width())
        _, context["var_buffer_size_multiplier"] = imgui.input_text("##buffer_size_multiplier_input_text",
                                                                    context["var_buffer_size_multiplier"],
                                                                    imgui.InputTextFlags_.chars_decimal)
        imgui.end_table()
        if imgui.button("Set variables", imgui.ImVec2(imgui.get_window_size()[0], 0)):
            global FPS, FRAME_COUNT
            FPS = int(context["var_fps"])
            FRAME_COUNT = int(context["var_buffer_size_multiplier"])*FPS
            restartAnomalyDetection()
            
def main():
    hello_imgui.apply_theme(ALL_THEMES[context["current_theme_index"]])
    currentTheme()
    currentModel()
    currentWindow()
    currentVideoPath()
    startAnomalyDetection()
    processAnomalyDetection()
    if context["video_capture"] != None:
        showVideoOptions()
        showPredictionStat()
        showVariables()
    
if __name__ == "__main__":
    compileModel()
    immapp.run(
        gui_function=main,
        with_implot=True,
        window_title="Anomaly Detection & Classification GUI App",
        window_size=(500, 700),
        with_markdown=True,
    )
