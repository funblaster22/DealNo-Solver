#ifndef BOX_H
#define BOX_H

struct Box {
    Box(int cx, int cy, int w, int h) : _cx(cx), _cy(cy), _w(w), _h(h) {}

    int cx() {
        return _cx;
    }

    void cx(int cx) {
        _cx = cx;
    }

    int cy() {
        return _cy;
    }

    void cy(int cy) {
        _cy = cy;
    }

    int w() {
        return _w;
    }

    void w(int w) {
        _w = w;
    }

    int h() {
        return _h;
    }

    void h(int h) {
        _h = h;
    }

    int area() {
        return _w * _h;
    }

    void show() {
        // TODO
    }

    void translate(int dx, int dy) {
        _cx += dx;
        _cy += dy;
    }

private:
    int _cx, _cy, _w, _h;
};

#endif