from kivy.app import App
from kivy.properties import OptionProperty, NumericProperty, ListProperty, \
        BooleanProperty, StringProperty, ObjectProperty
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.widget import Widget
import numpy as np
from kivy.graphics import *
from kivy.uix.dropdown import DropDown
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from mmlp import mmlp
import matplotlib.pyplot as plt
from kivy.uix.image import Image
import pickle 
#from scipy.ndimage.filters import gaussian_filter
from scipy.misc import *
from kivy.graphics.texture import Texture
#from PIL import Image as Img
import seaborn as sns;sns.set()
from kivy.config import Config
Config.set('kivy', 'exit_on_escape', '0')


Builder.load_string('''

<LinePlayground>:

    canvas:
        Color:
            rgba: .4, .4, 1, .5
        Line:
            points: self.points
            joint: 'round'
            cap: 'round'
            width: 2
            
        Color:
            rgba: .8, .8, .8, 1.
        Line:
            points: self.points
        Color:
            rgba: .8, .8, .8, 1.
        Line:
            points: self.points2        
        Color:
            rgba: .0, 1, 1, .5
        Line:
            points: self.points2
            joint: 'round'
            cap: 'round'
            width: 2             
        Color:
            rgba: .8, .8, .8, 1.
        Line:
            points: self.points3        
        Color:
            rgba: 1, 1, 0, .5
        Line:
            points: self.points3
            joint: 'round'
            cap: 'round'
            width: 2             
        Color:
            rgba: .8, .8, .8, .8
        Line:
            points: self.pointsG
        Line:
            points: self.pointsG2
        Line:
            points: self.pointsG3
        

    canvas.before:

        Rectangle:
            source: 'background.png'
            pos: self.pos
            size: 10,10#self.size  

        Rectangle:
            pos: self.width*.52,self.height*.35 
            size: self.height*.6,self.height*.6  
            texture: root.TEX
    
    Label:
        text: root.drawdims
        pos: self.height*.6,self.height*.42 
        size: self.size
        font_color: .1, .1, .1, .8 
    
    Label:
        text: root.netloss
        pos: -20,root.losspos
        size: self.size
        font_color: .1, .1, .1, .8
    Label:
        text: ''#root.accu
        pos: -self.width*.3,-self.height*.18
        size: self.size
        font_size: '40dp'
        font_color: .1, .1, .1, .8
    Label:
        text: root.time
        pos: root.steppos-self.width*.5,self.height*.17
        size: self.size
        font_color: .1, .1, .1, .8
   
    BoxLayout:
        orientation: 'horizontal'
        padding: "5dp"
        spacing: "5dp"

        BoxLayout:
            orientation: 'vertical'

            BoxLayout:
                orientation: 'horizontal'                
                Label:
                    text: "                      "

            BoxLayout:
                orientation: 'horizontal'
                Label:
                    text: "                      "

            BoxLayout:
                orientation: 'horizontal'
                Button:
                    text: 'save IMG'
                    background_color: 1,1,1, 0.4
                    size_hint_y: None
                    height: '40dp'
                    on_release: root.save_HD()
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:
                    text: 'save IMG'
                    background_color: 1,1,1, 0.4
                    size_hint_y: None
                    height: '40dp'
                    on_release: root.save_HD()
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:
                    text: 'save IMG'
                    background_color: 1,1,1, 0.4
                    size_hint_y: None
                    height: '40dp'
                    on_release: root.save_HD()
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:
                    text: 'save IMG'
                    background_color: 1,1,1, 0.4
                    size_hint_y: None
                    height: '40dp'
                    on_release: root.save_HD()
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                
                
            BoxLayout:
                orientation: 'horizontal'
                canvas.before:
                    BorderImage:
                        source: 'background.png'
                        pos: self.pos
                        size: self.size
                BoxLayout:
                    orientation: 'vertical'
                    padding: "5dp"
                    spacing: "20dp"
                    Label:
                        text: "data"                                
                    Label:
                        text: "targets"                    
                    Label:
                        text: "Xv"                    
                    Label:
                        text: "Yv"
                    
                BoxLayout:
                    orientation: 'vertical'
                    padding: "1dp"
                    spacing: "1dp"
                    TextInput:
                        id: train_data
                        hint_text: "/data"
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.Ptrain = root.NP.load(str(self.text))
                            else:\
                                root.Ptrain = root.Pdef
                    TextInput:
                        id: train_targets
                        hint_text: "/data"
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.ttrain = root.NP.load(str(self.text))
                            else:\
                                root.ttrain = root.tdef
                    TextInput:
                        id: test_data
                        hint_text: "/data"
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.Ptest = root.NP.load(str(self.text))
                            else:\
                                root.Ptest = root.NP.array([])
                    
                    TextInput:
                        id: test_targets
                        hint_text: "/data"
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        #on_text_validate: root.ttest = root.NP.load(str(self.text))
                            #if self.text != '':\
                                #root.ttest = root.NP.load(str(self.text))
                            #else:\
                                #root.ttest = root.NP.array([])
                        on_text_validate: root.CMAPS = str(self.text)

                BoxLayout:
                    orientation: 'vertical'
                    padding: "5dp"
                    spacing: "10dp"
                    Label:
                        text: "IN"                    
                    
                    Button:                    
                        text: '1st'
                        background_color: 1,1, 1, .2
                        on_release: root.outstate = 'h1';root.p1 = root.draw_dream();root.pointsG3 = []
                        
                    Button:                    
                        text: '2nd'
                        background_color: 1,1, 1, .2
                        on_release: root.outstate = 'h2';root.p1 = root.draw_dream();root.pointsG3 = []
                        
                    Button:                    
                        text: '3rd'
                        background_color: 1,1, 1, .2
                        on_release: root.outstate = 'h3';root.p1 = root.draw_dream();root.pointsG3 = []
                    
                    Button:                    
                        text: 'out'
                        background_color: 1,1, 1, .2
                        on_release: root.outstate = 'out';root.p1 = root.draw_dream();root.pointsG3 = []
                        
                                             
                    
                    
                BoxLayout:
                    orientation: 'vertical'
                    padding: "1dp"
                    spacing: "1dp"
                    TextInput:
                        id: in_
                        hint_text: root.inp
                        text: root.inp
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.in_dim = int(self.text);self.hint_text = str(root.network.in_dim);root.network.reset();root.scidata = []
                            else:\
                                self.text = self.hint_text
                                    
                    
                    TextInput:
                        id: h1_
                        hint_text: root.hid1
                        text: root.hid1
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.h1 = int(self.text);self.hint_text = str(root.network.h1);root.network.reset();root.scidata = []
                            else:\
                                self.text = self.hint_text
                    TextInput:
                        id: h2_
                        hint_text: root.hid2
                        text: root.hid2
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.h2 = int(self.text);self.hint_text = str(root.network.h2);root.network.reset();root.scidata = []
                            else:\
                                self.text = self.hint_text
                    TextInput:
                        id: h3_
                        hint_text: root.hid3
                        text: root.hid3
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.h3 = int(self.text);self.hint_text = str(root.network.h3);root.network.reset();root.scidata = []
                            else:\
                                self.text = self.hint_text

                    TextInput:
                        id: out_
                        hint_text: root.outp
                        text: root.outp
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.out = int(self.text);self.hint_text = str(root.network.out);root.network.reset();root.setup_train();root.scidata = []
                            else:\
                                self.text = self.hint_text

                BoxLayout:
                    orientation: 'vertical'
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                    padding: "1dp"
                    spacing: "1dp"
                    Label:
                        text: "transfer"                    
                    Label:
                        text: "activation"                    
                    Label:
                        text: "loss"
                                        
                BoxLayout:
                    orientation: 'vertical'
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                    padding: "1dp"
                    spacing: "1dp"
                    Button:
                        id: trnf
                        text: str(root.network.f)[9:16]
                        background_color: 1,1, 1, .2
                        foreground_color: 1, 1, 1, 1
                        on_release: dropdownTrnf.open(self);self.text = str(root.network.f)[9:16]
                                        
                    Button:
                        id: actv
                        text: str(root.network.f2)[9:16]
                        background_color: 1,1, 1, .2
                        foreground_color: 1, 1, 1, 1
                        on_release: dropdownAct.open(self);self.text = str(root.network.f2)[9:16]
                                            
                    Button:
                        id: lss
                        text: str(root.network.err)[9:14]
                        background_color: 1,1, 1, .2
                        foreground_color: 1, 1, 1, 1
                        on_release: dropdownLoss.open(self);self.text = str(root.network.err)[9:14]
                        
                    DropDown:
                        id: dropdownLoss
                        on_select: lss.text = '{}'.format(args[1]); self.dismiss();

                        Button:
                            text: 'quadratic'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownLoss.select('quadratic');root.network.err  = root.MM.qef
                        Button:
                            text: 'pseudo huber'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownLoss.select('pseudo huber');root.network.err  = root.MM.phl
                        Button:
                            text: 'binary cross entropy'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownLoss.select('BCE');root.network.err  = root.MM.bce
                    
                    DropDown:
                        id: dropdownAct
                        on_select: actv.text = '{}'.format(args[1]);self.dismiss();
                        
                        Button:
                            text: 'tanh'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('tanh');root.network.f2  = root.MM.f_tanh
                        Button:
                            text: 'logistic'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('logistic');root.network.f2  = root.MM.f_lgtr
                        Button:
                            text: 'relu'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('relu');root.network.f2  = root.MM.f_relu
                        Button:
                            text: 'identity'
                            size_hint_y: None
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            height: '30dp'
                            on_release: dropdownAct.select('identity');root.network.f2  = root.MM.f_iden
                        Button:
                            text: 'atan'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('atan');root.network.f2  = root.MM.f_atan
                        Button:
                            text: 'bent ident'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('bent ident');root.network.f2  = root.MM.f_bi                        
                        Button:
                            text: 'softplus'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('softplus');root.network.f2  = root.MM.f_sp
                        Button:
                            text: 'stochastic'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('stochastic');root.network.f2  = root.MM.f_stoch
                        Button:
                            text: 'binary'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownAct.select('binary');root.network.f2  = root.MM.f_bin
                    
                    DropDown:

                        id: dropdownTrnf
                        
                        on_select: trnf.text = '{}'.format(args[1]);self.dismiss();

                        Button:
                            text: 'tanh'
                            size_hint_y: None
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            height: '30dp'
                            on_release: dropdownTrnf.select('tanh');root.network.f  = root.MM.f_tanh
                        Button:
                            text: 'logistic'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('logistic');root.network.f  = root.MM.f_lgtr
                        Button:
                            text: 'relu'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('relu');root.network.f  = root.MM.f_relu
                        Button:
                            text: 'identity'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('identity');root.network.f  = root.MM.f_iden
                        Button:
                            text: 'atan'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('atan');root.network.f  = root.MM.f_atan
                        Button:
                            text: 'bent ident'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('bent ident');root.network.f  = root.MM.f_bi
                        Button:
                            text: 'softplus'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('softplus');root.network.f  = root.MM.f_sp
                        Button:
                            text: 'stochastic'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('stochastic');root.network.f = root.MM.f_stoch
                        Button:
                            text: 'binary'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownTrnf.select('binary');root.network.f  = root.MM.f_bin
                            
        BoxLayout:
            orientation: 'vertical'

            BoxLayout:
                orientation: 'horizontal'
                Label:
                    text: "                      "

            BoxLayout:
                orientation: 'horizontal'
                Label:
                    text: "                      "

            BoxLayout:
                orientation: 'horizontal'
                ToggleButton: 
                    id: training                   
                    text: 'Train'
                    size_hint_y: None
                    height: '40dp'
                    background_color: 1,1,1, 0.4
                    foreground_color: 1, 1, 1, 1
                    on_state: root.animate(self.state == 'down')
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                ToggleButton:
                    text: 'Dream'
                    size_hint_y: None
                    height: '40dp'
                    background_color: 1,1,1, 0.4
                    foreground_color: 1, 1, 1, 1
                    on_state: root.dream = (self.state == 'down')
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:                    
                    text: 'Reset'
                    size_hint_y: None
                    height: '40dp'
                    background_color: 1,1,1, 0.4
                    foreground_color: 1, 1, 1, 1
                    on_press: root.points = root.points2 = root.points3 = root.pointsG3 = root.review_arr = [];\
                     root.network.reset(); root.errors = root.errors_test = root.zero_arr; root.time = root.netloss = '';\
                     root.TEX.blit_buffer(root.zeros, bufferfmt="ubyte", colorfmt="rgba");\
                     root.scidata = []
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:
                    text: 'Save'
                    size_hint_y: None
                    height: '40dp'
                    background_color: 1,1,1, 0.4
                    foreground_color: 1, 1, 1, 1
                    on_press: root.pk.dump( root.network, open( "model.p", "wb" ) )
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size
                Button:
                    text: 'Load'
                    size_hint_y: None
                    height: '40dp'
                    background_color: 1,1,1, 0.4
                    foreground_color: 1, 1, 1, 1
                    on_press: root.network = root.pk.load(open( "model.p", "rb" ));root.load_text()
                    canvas.before:
                        BorderImage:
                            source: 'background.png'
                            pos: self.pos
                            size: self.size

            BoxLayout:
                orientation: 'horizontal'
                canvas.before:
                    BorderImage:
                        source: 'background.png'
                        pos: self.pos
                        size: self.size
                BoxLayout:
                    orientation: 'vertical'
                    padding: "5dp"
                    spacing: "10dp"
                    Label:
                        text: "optimizer"
                    Label:
                        text: "learning rate"
                    Label:
                        text: "b1"                    
                    Label:
                        text: "b2"                    
                    Label:
                        text: "eps"
                    
                BoxLayout:
                    orientation: 'vertical'
                    padding: "1dp"
                    spacing: "1dp"
                    Button:
                        id: optm
                        background_color: 1,1, 1, .2
                        foreground_color: 1, 1, 1, 1
                        text: str(root.network.optimizer)
                        on_release: dropdownOpt.open(self)

                    TextInput:
                        id: lr_
                        hint_text: str(root.network.eta)
                        text: root.eta
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.eta = float(self.text);self.hint_text = str(root.network.eta)
                            else:\
                                self.text = self.hint_text

                    TextInput:
                        id: b1_
                        text: root.beta1
                        hint_text: str(root.network.beta1)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.beta1  = float(self.text);self.hint_text = str(root.network.beta1)
                            else:\
                                self.text = self.hint_text

                    
                    TextInput:
                        id: b2_
                        text: root.beta2
                        hint_text: str(root.network.beta2)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.beta2  = float(self.text);self.hint_text = str(root.network.beta2)
                            else:\
                                self.text = self.hint_text


                    TextInput:
                        id: eps_
                        text: str(root.eps)
                        hint_text: str(root.eps)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.eps = self.text ;self.hint_text = self.text; root.network.eps = float(self.text)
                            else:\
                                self.text = self.hint_text

    
                    DropDown:
                        id: dropdownOpt
                        on_select: root.network.optimizer = root.optimizer  = optm.text = '{}'.format(args[1]);self.dismiss()

                        Button:
                            text: 'Adam'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            
                            on_release: dropdownOpt.select('Adam')
                        Button:
                            text: 'RMSprop'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            
                            on_release: dropdownOpt.select('RMSprop')
                        Button:
                            text: 'normal'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownOpt.select('normal')

                BoxLayout:
                    orientation: 'vertical'
                    padding: "1dp"
                    spacing: "1dp"
                    Label:
                        text: "regularizer"
                    Label:
                        text: "lambda"
                    Label:
                        text: "batch size"
                    Label:
                        text: "input droptout"                    
                    Label:
                        text: "layer dropout"
                  
                BoxLayout:
                    orientation: 'vertical'
                    padding: "1dp"
                    spacing: "1dp"
                    Button:
                        id: rgrz
                        text: str(root.network.reg)[9:13]
                        background_color: 1,1, 1, .2
                        foreground_color: 1, 1, 1, 1
                        on_release: dropdownReg.open(self)

                    TextInput:
                        id: lam_
                        text: root.lam
                        hint_text: str(root.network.lam)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.lam  = float(self.text)
                            else:\
                                self.text = self.hint_text

                    TextInput:
                        id: batch_
                        text: root.batch
                        hint_text: '64'
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.batch = self.text
                            else:\
                                self.text = self.hint_text

                    TextInput:
                        id: drop1_
                        text: root.drop1
                        hint_text: str(root.network.drop1)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.drop1  = float(self.text)
                            else:\
                                self.text = self.hint_text
                    
                    TextInput:
                        id: drop2_
                        text: root.drop2
                        hint_text: str(root.network.drop2)
                        background_color: 0, 0, 0, 0
                        cursor_color: 1, 1, 1, 1
                        foreground_color: 1, 1, 1, 1
                        multiline: False
                        on_text_validate: 
                            if self.text != '':\
                                root.network.drop2 = float(self.text)
                            else:\
                                self.text = self.hint_text

                
                    DropDown:
                        id: dropdownReg
                        on_select: self.dismiss()

                        Button:
                            text: 'L1'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            size_hint_y: None
                            height: '30dp'
                            on_release: dropdownReg.select('L1');rgrz.text = 'L1';root.network.reg = root.MM.L1

                        Button:
                            text: 'L2'
                            size_hint_y: None
                            height: '30dp'
                            background_color: 0.2, .2,.2, 1
                            foreground_color: 1, 1, 1, 1
                            on_release: dropdownReg.select('L2');rgrz.text = 'L2';root.network.reg = root.MM.L2
                        
''')


class LinePlayground(FloatLayout):
    
    NP = np
    MM = mmlp
    pk = pickle
    
    R = np.linspace(-2, 2, 100, endpoint=True) 
    A,B = np.meshgrid(R,R)
    G = [] 
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            G += [[A[i][i],A[i][j]]]
    G = np.array(G)
    R = np.linspace(-2, 2, 1000, endpoint=True)
    A,B = np.meshgrid(R,R)
    Gmax = [] 
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Gmax += [[A[i][i],A[i][j]]]
    Gmax = np.array(Gmax)
    ###################### XOR data (with thrid class in the center) ################
    p1 = np.zeros([120,2])
    p2 = np.zeros([120,2])
    p3 = np.zeros([240,2])
    mu1 = np.array([-.75,.75])
    mu2 = np.array([.75,-.75])
    mu3 = np.array([.75,.75])
    mu4 = np.array([-.75,-.75])
    var = np.diag([0.1,0.1])
    choice1 = np.random.rand(120)
    choice2 = np.random.rand(120)
    for i in range(p1.shape[0]):
        if choice1[i]>.5 :
            p1[i] = np.random.multivariate_normal(mu1,var)
        else:
            p1[i] = np.random.multivariate_normal(mu2,var)
        if choice2[i]>.5 :
            p2[i] = np.random.multivariate_normal(mu3,var)
        else:
            p2[i] = np.random.multivariate_normal(mu4,var)
    for i in range(p3.shape[0]):       
        p3[i] = np.random.multivariate_normal([0,0],var)
    Pdef = np.concatenate((p1,p2,p3),0) 
    tdef = np.zeros(480)
    tdef[120:240] = 1
    tdef[240:] = 2
    P = np.copy(Pdef)
    t = np.copy(tdef)
    for i in range(p1.shape[0]):
        if choice1[i]>.5 :
            p1[i] = np.random.multivariate_normal(mu1,var)
        else:
            p1[i] = np.random.multivariate_normal(mu2,var)
        if choice2[i]>.5 :
            p2[i] = np.random.multivariate_normal(mu3,var)
        else:
            p2[i] = np.random.multivariate_normal(mu4,var)
    for i in range(p3.shape[0]):       
        p3[i] = np.random.multivariate_normal([0,0],var)
    Ptest = np.concatenate((p1,p2,p3),0) 
    ttest = tdef
    ########################################################
    Ptrain = P
    ttrain = t

    inp = StringProperty('2')
    hid1 = StringProperty('64')
    hid2 = StringProperty('32')
    hid3 = StringProperty('16')
    outp = StringProperty('1')

    optimizer = StringProperty('Adam')    
    beta1 = StringProperty('0.85')         # smoothing parameter (Adam)
    beta2 = StringProperty('0.9')        # decay rate (RMSprop,Adam)
    eta = StringProperty('1e-2')         # initial learning rate
    lam = StringProperty('1e-5')        # lambda for reguarization
    batch = StringProperty('64')
    
    drop1 = StringProperty('1.')
    drop2 = StringProperty('1.')
    eps = StringProperty('1e-7')
    

    network = mmlp.mlp(in_dim=2,h1=64,h2=32,h3=16,out=1)
    errors = np.array([0.])
    errors_test = np.array([0.])
    netloss = StringProperty('')
    time = StringProperty('')
    accu = StringProperty('Accuracy')
    
    losspos = NumericProperty(0.5)
    steppos = NumericProperty(0.5)
    drawdims = StringProperty('')
    
    zeros = (np.zeros((100,100,4)))
    zeros[0] = -np.ones(zeros[0].shape)
    zeros[-1] = -np.ones(zeros[0].shape)
    zeros[:,0] = -np.ones(zeros[:,0].shape)
    zeros[:,-1] = -np.ones(zeros[:,0].shape)
    zeros = zeros.astype(np.uint8).tostring()
    review_arr = []

    zero_arr = np.array([0.])
     
    
    points = ListProperty([])
    points2 = ListProperty([])
    points3 = ListProperty([])
    pointsG = ListProperty([])
    pointsG2 = ListProperty([])
    pointsG3 = ListProperty([])
    points3 = ListProperty([])

    _update_points_animation_ev = None
    
    TEX = Texture.create(size=(100, 100), colorfmt="rgba")
    TEX.blit_buffer(zeros, bufferfmt="ubyte", colorfmt="rgba")
    dream = False
    seeds = int(np.random.randint(1,100))
    onehot = BooleanProperty(False)
    outstate = StringProperty('out')
    drawdata = np.empty((1,3))
    CMAPS = 'gist_ncar_r'
    scidata = []
    def draw_graphs(self):
        self.pointsG = [(self.width * 0.03,self.height * 0.9),
                                    [self.width * 0.03,self.height * 0.7],
                                    [self.width * 0.45,self.height * 0.7],
                                    ]
        self.pointsG2 = [(self.width * 0.03,self.height * 0.6),
                                    [self.width * 0.03,self.height * 0.4],
                                    [self.width * 0.45,self.height * 0.4],
                                    ]
    

    def draw_dream(self):
        
        arr_out = self.network.predict(self.G)
        visu = None
        frame = np.zeros((100,100)) 

        if self.outstate == 'h1' and self.network.h1 != 0:
            visu = self.network.s1
        if self.outstate == 'h2' and self.network.h2 != 0:
            visu = self.network.s2
        if self.outstate == 'h3' and self.network.h3 != 0:
            visu = self.network.s3
        if self.outstate != 'out' and visu is not None:
            #np.random.seed(self.seeds)
            #ind = np.random.choice(np.arange(visu.shape[0]),int(np.min([visu.shape[0],16])),replace=False)
            #print(ind)
            
            ind = np.arange(visu.shape[0])
            if visu.shape[0] > 27:
                ind = np.append(ind[:8],ind[-9:-1])
            ind = ind[:16]
            j = 0
            for i in ind:
                if j < 4:
                    frame[:25,25*j:25*j+25] = visu[i].reshape(100,100)[::4,::4]
                elif j < 8:
                    frame[25:50,25*(j-4):25*(j-4)+25] = visu[i].reshape(100,100)[::4,::4]
                elif j < 12:
                    frame[50:75,25*(j-8):25*(j-8)+25] = visu[i].reshape(100,100)[::4,::4]
                elif j < 16:
                    frame[75:,25*(j-12):25*(j-12)+25] = visu[i].reshape(100,100)[::4,::4]
                j += 1
            arr = frame                        
            arr = arr.reshape(-1)

        else:
            arr = arr_out
            
        if self.onehot and visu is None:
            #tar = np.array([np.argmax(x) for x in y])
            #arr2 = self.network.predict(x1)
            #arr2 = np.array([np.argmax(x) for x in arr2])
            arr = np.array([np.argmax(x)+np.max(x) for x in arr])
            #self.accu = '{:.4f} % '.format(np.where((arr2-tar) == 0)[0].shape[0]*100. / tar.shape[0])

        
        if np.max(np.abs(arr)) != 0.: arr = arr /np.max(np.abs(arr))
        arr = arr.reshape(100,100)#.T
        tranfer = plt.get_cmap(self.CMAPS)
        arr = tranfer(arr)
        arr[:,:,3] *= 255 # alpha
        arr[:,:,:3] *= 255 # brightness
        arr = arr.astype(np.uint8)
        data = arr.tostring()
        
        self.TEX.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgba")
        
        return data


    def draw_loss(self):
        cy = self.height * 0.9  #starting y
        cx = self.width * 0.03  #starting x
        w = self.width * 0.4    #width
        err = np.copy(self.errors)
        err2 = np.copy(self.errors)
        err2 = err2[-int(w):]
        
        if err.shape[0] > w:
            err = err[::np.max([1,int(err.shape[0] / (w))])]
        else:
            err = err[::1]
        err = err[:int(w)]
        
        if self.Ptest != [] and self.network.count > 0:
            fehler = self.network.predict(self.Ptest) 
            testloss = self.network.err(self.ttest,fehler)     
            self.errors_test = np.append(self.errors_test[-int(w)-10:],testloss)
            testerr = np.copy(self.errors_test)
            testerr = testerr[-int(w):]

        if np.max(err) != 0.: err /= np.max(err)
        values = self.height * 0.7 + self.height * 0.2*err 
        scale = np.max(np.copy(err2)) 
        if self.Ptest != [] and self.network.count > 0:
            if np.max(testerr) > scale : scale = np.max(testerr)
        if np.max(err2) != 0.: err2 /= scale
        values2 = self.height * 0.4 + self.height * 0.2*err2
        if self.Ptest != [] and self.network.count > 0:
            if np.max(testerr) != 0.:testerr /= np.max(np.abs(scale))
            values3 = self.height * 0.4 + self.height * 0.2*testerr
        
        points = []
        points2 = []
        points3 = []

        self.netloss = str(self.errors[-1])[:6]
        self.losspos = int(values2[-1]-self.height*.5)
        self.steppos = int(cx+err.shape[0])
        self.time = str(self.network.count)

        for i in range(values.shape[0]):
            points.append(cx + i)
            points.append(values[i])

        for i in range(values2.shape[0]):
            points2.append(cx + i)
            points2.append(values2[i])
        if self.Ptest != [] and self.network.count > 0:
            for i in range(values3.shape[0]):
                points3.append(cx + i)
                points3.append(values3[i])

        self.points = points
        self.points2 = points2
        self.points3 = points3


    def setup_train(self):
        if self.drawdata.shape[0] < 20:
            self.P = self.Ptrain
            self.t = self.ttrain

        else:
            self.P = self.drawdata[1:,:2]
            self.t = self.drawdata[1:,2]                               
            for k in range(np.unique(self.t).shape[0]):
              self.t[np.where(self.t==np.unique(self.t)[k])[0]] = k
            

        if int(self.network.out) > 1: 
            self.onehot = True
        else:
            self.onehot = False

        if self.onehot and (len(self.t.shape)==1):
            iterate = np.unique(self.t)
            label = []
            for i in iterate:
                label += [np.where(self.t==i,1,0)]
            label = np.array(label).T
            self.t = label
            self.ttest = self.t
        

    def animate(self, do_animation):  
              
        if do_animation:
            self.drawdims = ''
            self.accu = ''
            self.pointsG3 = []
            self.setup_train()
            
            self._update_points_animation_ev = Clock.schedule_interval(
                self.update_points_animation, 0)
            
    


        elif self._update_points_animation_ev is not None:
            self._update_points_animation_ev.cancel()
            
    def load_text(self):
        
        self.inp = str(int(self.network.in_dim))
        self.hid1 = str(int(self.network.h1))
        self.hid2 = str(int(self.network.h2))
        self.hid3 = str(int(self.network.h3))
        self.outp = str(int(self.network.out))
        self.optimizer = str(self.network.optimizer)    
        self.beta1 = str(self.network.beta1)       
        self.beta2 = str(self.network.beta2)        
        self.drop1 = str(self.network.drop1)        
        self.drop2 = str(self.network.drop2)        
        self.eta = str(self.network.eta)        
        self.lam = str(self.network.lam)         
        self.eps = str(self.network.eps)


    def save_HD(self):        
        visu = None

        if self.outstate == 'h1' and self.network.h1 != 0:
            visu = self.network.s1
        if self.outstate == 'h2' and self.network.h2 != 0:
            visu = self.network.s2
        if self.outstate == 'h3' and self.network.h3 != 0:
            visu = self.network.s3
        
        if self.outstate != 'out' and visu is not None:
            arr_out = self.network.predict(self.G)
            frame = np.zeros(( (int(np.sqrt(visu.shape[0]))+1)*100,100*(int(np.sqrt(visu.shape[0]))+1)  ))
            for i in range(int(np.sqrt(visu.shape[0]))+1):
                for k in range(int(np.sqrt(visu.shape[0]))+1):
                    try:
                        frame[i*100:(i+1)*100,\
                                k*100:(k+1)*100] = visu[i*(int(np.sqrt(visu.shape[0]))+1)+k].reshape(100,100)#[::(int(np.sqrt(visu.shape[0]))+1),::(int(np.sqrt(visu.shape[0]))+1)]
                    except:
                        break;
            arr = frame                        
            arr = arr.reshape(-1)
            if np.max(np.abs(arr)) != 0. : arr = arr /np.max(np.abs(arr))
            arr = arr.reshape(frame.shape[0],frame.shape[1])

        else:

            arr = self.network.predict(self.Gmax)
            
            if self.onehot and visu is None:
                #tar = np.array([np.argmax(x) for x in y])
                #arr2 = self.network.predict(x1)
                #arr2 = np.array([np.argmax(x) for x in arr2])
                arr = np.array([np.argmax(x)+np.max(x) for x in arr])
                #self.accu = '{:.4f} % '.format(np.where((arr2-tar) == 0)[0].shape[0]*100. / tar.shape[0])
            if np.max(np.abs(arr)) != 0. : arr = arr /np.max(np.abs(arr))
            arr = arr.reshape(1000,1000)#.T
        '''
        fig = plt.figure(figsize=(10,8))       
        sns.heatmap(np.flip(arr,0),xticklabels=False, yticklabels=False,cmap=self.CMAPS, vmin=0, vmax=1)
        fig.show()
        
        fig2 = plt.figure(figsize=(10,8)) 
        #ax = fig2.add_subplot(111)      
        sns.tsplot(np.array(np.array(self.scidata)[:,0].tolist()).T,err_style="unit_traces",color='orange')
        sns.tsplot(np.array(np.array(self.scidata)[:,1].tolist()).T,err_style="unit_traces",color='blue')
        sns.tsplot(np.array(np.array(self.scidata)[:,2].tolist()).T,err_style="unit_traces",color='green')
        sns.tsplot(np.array(np.array(self.scidata)[:,3].tolist()).T,err_style="unit_traces",color='red')
        #plt.legend()
        fig2.show()
        '''
        tranfer = plt.get_cmap(self.CMAPS)
        arr = tranfer(arr)
        arr[:,:,3] *= 255 # alpha
        arr[:,:,:3] *= 255 # brightness
        imsave((self.outstate+'.png'),np.flip(arr,0))
        print(self.outstate+' image saved.')

    def update_points_animation(self,dt):
        self.load_text()
        w = self.width * 0.4    #width
        if int(self.batch) != 0:
            j = np.random.choice(range(self.P.shape[0]),int(self.batch),replace=False)
            x1 = self.P[j]
            y = self.t[j]
        else:
            x1 = self.P
            y = self.t

        try :
            self.network.train(x1,y)
        except ValueError:
            print('Duuuuude, network and data dimensions should match, you know?')
            print('Wooooaa maaaan, dont put that data in there, dimensions colliding')
            print('Great, you broke it...')
            self._update_points_animation_ev.cancel()
            return -1
        '''
        #print(np.linalg.norm(self.network.w.T,axis=0).shape)
        if self.hid2 == '0' and self.hid3 == '0':
            self.scidata += [np.array([np.linalg.norm(self.network.w.T,axis=0),np.linalg.norm(self.network.v.T,axis=0)])]
            print('h1')
        elif self.hid3 == '0':
            self.scidata += [np.array([np.linalg.norm(self.network.w.T,axis=0),np.linalg.norm(self.network.v.T,axis=0),np.linalg.norm(self.network.u.T,axis=0)])]
            print('h1,h2')
        else: 
            self.scidata += [np.array([np.linalg.norm(self.network.w.T,axis=0),np.linalg.norm(self.network.v.T,axis=0),np.linalg.norm(self.network.u.T,axis=0),np.linalg.norm(self.network.z.T,axis=0)])]
            print('h1,h2,h3')
        print(np.array(np.array(self.scidata).tolist()).shape)
        '''

        self.errors = np.append(self.errors,(self.network.loss))
        self.draw_loss()
        self.draw_graphs()

        if self.inp == '2' and self.dream:
            data = self.draw_dream()
            self.review_arr += [data]
            self.review_arr = self.review_arr[-int(w):]
        else:
            data = self.zeros
            self.review_arr += [data]
            self.review_arr = self.review_arr[-int(w):] 

    def on_touch_down(self, touch):
        self.draw_loss()
        d = 4.
        if (touch.x - d / 2 > self.width*.52) and (touch.x - d / 2 < self.width*.52+self.height*.6) and (self.ids['training'].state != 'down' ):
            if (touch.y - d / 2 > self.height*.35) and (touch.y - d / 2 < self.height*.35+self.height*.6):
                with self.canvas.after:
                    Color(np.random.uniform(0.,1),np.random.uniform(0.,1),np.random.uniform(0.,1),.5)
                    Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
                    self.Ptest = np.array([])
        if touch.is_double_tap  and (self.ids['training'].state != 'down' ):
                self.canvas.after.clear()
                self.drawdata = np.empty((1,3))
                self.drawdims = ''
                #self.drawdata = np.delete(self.drawdata,0)
        super(LinePlayground, self).on_touch_down(touch)
    def on_touch_up(self,touch):
        self.draw_graphs()
        self.setup_train()
        super(LinePlayground, self).on_touch_up(touch)
    def on_touch_move(self, touch):
        #### display history
        if (touch.x  >= self.width*.03) and (touch.x  < self.width*.43) and (touch.x-self.width*.03 < len(self.review_arr)):
            if (touch.y  > self.height*.4) and (touch.y  < self.height*.6):
                self.TEX.blit_buffer(self.review_arr[int(touch.x-self.width*.03)], bufferfmt="ubyte", colorfmt="rgba")
                if int(self.time) > self.width*.4: ##### häää??!!
                   self.netloss = str('{}'.format(int(int(self.time)-self.width*.4 +int(touch.x-self.width*.03)+1)))
                else:
                   self.netloss = str('{}'.format(int(touch.x-self.width*.03)+1))
                self.pointsG3 = [[int(touch.x),self.height*.65],[touch.x,self.height*.35]]

        d = 4.
        try: 
            startX = self.drawdata[-1][0]
        except:
            startX = 0
        try: 
            startY = self.drawdata[-1][1]
        except:
            startY = 0

        if (touch.x - d / 2 > self.width*.52) and (touch.x - d / 2 < self.width*.52+self.height*.6)  and (self.ids['training'].state != 'down' ):
            if (touch.y - d / 2 > self.height*.35) and (touch.y - d / 2 < self.height*.35+self.height*.6):
                if np.linalg.norm([touch.x - startX,touch.y - startY]) > self.height*0.01: 
                    with self.canvas.after:
                        Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d),alpha=.5)
                    
                    x = (int(touch.x*1.)-int(1.*self.width*.52+self.height*.3))*1. / int(self.height*.15)*1.
                    y = (int(touch.y*1.)-int(1.*self.height*.35+self.height*.3))*1. / int(self.height*.15)*1.
                    #print('touches',x,y)
                    self.drawdata = np.vstack((self.drawdata,[y,x,touch.uid]))                    
                    #print(self.drawdata.shape)  
                    self.drawdims = str(np.unique(self.drawdata[1:,2]).shape[0])
        super(LinePlayground, self).on_touch_move(touch) 

class MML(App):
    
    def build(self):
        L = LinePlayground()   
        #some bug has dropdown open on start..
        L.ids['dropdownTrnf'].dismiss()
        L.ids['dropdownAct'].dismiss()
        L.ids['dropdownLoss'].dismiss()
        L.ids['dropdownReg'].dismiss()
        L.ids['dropdownOpt'].dismiss()
        return L

if __name__ == '__main__':
    MML().run()