# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/calculator_options.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/framework/calculator_options.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_pb=_b('\n,mediapipe/framework/calculator_options.proto\x12\tmediapipe\"9\n\x11\x43\x61lculatorOptions\x12\x18\n\x0cmerge_fields\x18\x01 \x01(\x08\x42\x02\x18\x01*\n\x08\xa0\x9c\x01\x10\x80\x80\x80\x80\x02\x42\x34\n\x1a\x63om.google.mediapipe.protoB\x16\x43\x61lculatorOptionsProto')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_CALCULATOROPTIONS = _descriptor.Descriptor(
  name='CalculatorOptions',
  full_name='mediapipe.CalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='merge_fields', full_name='mediapipe.CalculatorOptions.merge_fields', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\030\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(20000, 536870912), ],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=116,
)

DESCRIPTOR.message_types_by_name['CalculatorOptions'] = _CALCULATOROPTIONS

CalculatorOptions = _reflection.GeneratedProtocolMessageType('CalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _CALCULATOROPTIONS,
  __module__ = 'mediapipe.framework.calculator_options_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.CalculatorOptions)
  ))
_sym_db.RegisterMessage(CalculatorOptions)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\032com.google.mediapipe.protoB\026CalculatorOptionsProto'))
_CALCULATOROPTIONS.fields_by_name['merge_fields'].has_options = True
_CALCULATOROPTIONS.fields_by_name['merge_fields']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\030\001'))
# @@protoc_insertion_point(module_scope)
