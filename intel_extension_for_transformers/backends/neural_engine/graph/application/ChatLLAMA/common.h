// Various helper functions and utilities

#pragma once

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>

#if !defined (_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

//
// Console utils
//

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

enum console_color_t {
    CONSOLE_COLOR_DEFAULT=0,
    CONSOLE_COLOR_PROMPT,
    CONSOLE_COLOR_USER_INPUT
};

struct console_state {
    bool multiline_input = false;
    bool use_color = false;
    console_color_t color = CONSOLE_COLOR_DEFAULT;

    FILE* out = stdout;
#if defined (_WIN32)
    void* hConsole;
#else
    FILE* tty = nullptr;
    termios prev_state;
#endif
};

void console_init(console_state & con_st);
void console_cleanup(console_state & con_st);
void console_set_color(console_state & con_st, console_color_t color);
bool console_readline(console_state & con_st, std::string & line);
