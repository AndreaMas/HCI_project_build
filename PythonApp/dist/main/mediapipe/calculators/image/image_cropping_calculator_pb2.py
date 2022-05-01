# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/image/image_cropping_calculator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/calculators/image/image_cropping_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_pb=_b('\n;mediapipe/calculators/image/image_cropping_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xe4\x03\n\x1eImageCroppingCalculatorOptions\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x13\n\x08rotation\x18\x03 \x01(\x02:\x01\x30\x12\x12\n\nnorm_width\x18\x04 \x01(\x02\x12\x13\n\x0bnorm_height\x18\x05 \x01(\x02\x12\x18\n\rnorm_center_x\x18\x06 \x01(\x02:\x01\x30\x12\x18\n\rnorm_center_y\x18\x07 \x01(\x02:\x01\x30\x12V\n\x0b\x62order_mode\x18\x08 \x01(\x0e\x32\x34.mediapipe.ImageCroppingCalculatorOptions.BorderMode:\x0b\x42ORDER_ZERO\x12\x18\n\x10output_max_width\x18\t \x01(\x05\x12\x19\n\x11output_max_height\x18\n \x01(\x05\"K\n\nBorderMode\x12\x16\n\x12\x42ORDER_UNSPECIFIED\x10\x00\x12\x0f\n\x0b\x42ORDER_ZERO\x10\x01\x12\x14\n\x10\x42ORDER_REPLICATE\x10\x02\x32W\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xdf\xd6\x93} \x01(\x0b\x32).mediapipe.ImageCroppingCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_IMAGECROPPINGCALCULATOROPTIONS_BORDERMODE = _descriptor.EnumDescriptor(
  name='BorderMode',
  full_name='mediapipe.ImageCroppingCalculatorOptions.BorderMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BORDER_UNSPECIFIED', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BORDER_ZERO', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BORDER_REPLICATE', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=433,
  serialized_end=508,
)
_sym_db.RegisterEnumDescriptor(_IMAGECROPPINGCALCULATOROPTIONS_BORDERMODE)


_IMAGECROPPINGCALCULATOROPTIONS = _descriptor.Descriptor(
  name='ImageCroppingCalculatorOptions',
  full_name='mediapipe.ImageCroppingCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='mediapipe.ImageCroppingCalculatorOptions.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='mediapipe.ImageCroppingCalculatorOptions.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='mediapipe.ImageCroppingCalculatorOptions.rotation', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='norm_width', full_name='mediapipe.ImageCroppingCalculatorOptions.norm_width', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='norm_height', full_name='mediapipe.ImageCroppingCalculatorOptions.norm_height', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='norm_center_x', full_name='mediapipe.ImageCroppingCalculatorOptions.norm_center_x', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='norm_center_y', full_name='mediapipe.ImageCroppingCalculatorOptions.norm_center_y', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='border_mode', full_name='mediapipe.ImageCroppingCalculatorOptions.border_mode', index=7,
      number=8, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_max_width', full_name='mediapipe.ImageCroppingCalculatorOptions.output_max_width', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_max_height', full_name='mediapipe.ImageCroppingCalculatorOptions.output_max_height', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.ImageCroppingCalculatorOptions.ext', index=0,
      number=262466399, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      options=None),
  ],
  nested_types=[],
  enum_types=[
    _IMAGECROPPINGCALCULATOROPTIONS_BORDERMODE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=113,
  serialized_end=597,
)

_IMAGECROPPINGCALCULATOROPTIONS.fields_by_name['border_mode'].enum_type = _IMAGECROPPINGCALCULATOROPTIONS_BORDERMODE
_IMAGECROPPINGCALCULATOROPTIONS_BORDERMODE.containing_type = _IMAGECROPPINGCALCULATOROPTIONS
DESCRIPTOR.message_types_by_name['ImageCroppingCalculatorOptions'] = _IMAGECROPPINGCALCULATOROPTIONS

ImageCroppingCalculatorOptions = _reflection.GeneratedProtocolMessageType('ImageCroppingCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _IMAGECROPPINGCALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.image.image_cropping_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.ImageCroppingCalculatorOptions)
  ))
_sym_db.RegisterMessage(ImageCroppingCalculatorOptions)

_IMAGECROPPINGCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _IMAGECROPPINGCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_IMAGECROPPINGCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
