#ifndef SFW_COMPAT_H
#define SFW_COMPAT_H

class ClassDB {
public:
  static void bind_method(...) {}
};

#define GDCLASS(m_class, m_inherits) SFW_OBJECT(m_class, m_inherits);

#define VARIANT_ENUM_CAST(...) 

#define BIND_ENUM_CONSTANT(...) 

#define D_METHOD(...) ""

#define PLOG_MSG LOG_MSG

#define PropertyInfo(...) ""
#define ADD_PROPERTY(...) 


#endif