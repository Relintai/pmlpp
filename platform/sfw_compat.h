#ifndef SFW_COMPAT_H
#define SFW_COMPAT_H

class ClassDB {
public:
  static void bind_method(...) {}
};

#define GDCLASS SFW_OBJECT

#define VARIANT_ENUM_CAST(...) 

#define BIND_ENUM_CONSTANT(...) 

#define D_METHOD(...) ""

#define PLOG_MSG LOG_MSG
#define PLOG_TRACE LOG_TRACE
#define PLOG_ERR LOG_ERR

#define PropertyInfo(...) ""
#define ADD_PROPERTY(...) 

#define ADD_GROUP(a, b) 

#endif