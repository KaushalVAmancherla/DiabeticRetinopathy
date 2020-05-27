from ._anvil_designer import Form1Template
from anvil import *
import anvil.server

class Form1(Form1Template):

  def __init__(self, **properties):
    self.init_components(**properties)
    
    
  def file_loader_1_change(self, file, **event_args):
    """This method is called when a new file is loaded into this FileLoader"""
    result, score = anvil.server.call('classify_image', file)
    
    self.result_lbl.text = "%s (%0.2f)" % (result, score)
    self.image_1.source = file

 