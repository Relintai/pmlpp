#include "gdexample.h"

void GDExample::_register_methods() {
	register_method("_process", &GDExample::_process);
	register_property<GDExample, float>("amplitude", &GDExample::amplitude, 10.0);
	register_property<GDExample, float>("speed", &GDExample::set_speed, &GDExample::get_speed, 1.0);

	//register_signal<GDExample>((char *)"position_changed", "node", PANDEMONIUM_VARIANT_TYPE_OBJECT, "new_pos", PANDEMONIUM_VARIANT_TYPE_VECTOR2);
	register_signal<GDExample>("position_changed");
}

GDExample::GDExample() {
}

GDExample::~GDExample() {
	// add your cleanup here
}

void GDExample::_init() {
	// initialize any variables here
	time_passed = 0.0;
	amplitude = 10.0;
	speed = 1.0;
}

void GDExample::_process(float delta) {
	time_passed += speed * delta;

	Vector2 pos = get_position();

	Vector2 new_position = Vector2(
			amplitude * sin(time_passed * speed),
			amplitude * cos(time_passed * speed));

	set_position(new_position);

	time_emit += delta;
	if (time_emit > 1.0) {
		emit_signal("position_changed");

		time_emit = 0.0;
	}
}

void GDExample::set_speed(float p_speed) {
	speed = p_speed;
}

float GDExample::get_speed() {
	return speed;
}
