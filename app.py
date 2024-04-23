import gradio as gr


description = ''' This is a Demo of our COMS 4995 Applied Computer Vision semester project: an ensemble method classification tool for EEG diagnoses.
Drag and drop an image or eeg parquet file and (hopefully) you'll get your result 

Authors: Raman Odgers, Akhil Golla, Vinayak Kannan, Sohan Kshirsagar

'''



def greet(image, dist, choice):
    

    return ('gingus')


demo = gr.Interface(
    fn=greet,
    inputs=[gr.Image(), 
            gr.Slider(0, 10, step = 1, label = "Error Distance"), 
            gr.Radio(["Big CLDPC", "Small CLDPC","Surface 25","Steane 17"], label="Parity-Check matrix", info="What matrix should we use?", value = 'Big CLDPC')],

    outputs=[gr.Textbox(label="EEG class")],

    title="EEG Classification (APPLIED CV COMS 4995)",
    description = description, 
    live = True

)
demo.launch( share = False)