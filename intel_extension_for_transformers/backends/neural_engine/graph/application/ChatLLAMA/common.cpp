//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include "common.h"

#include <cassert>
#include <iostream>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <unordered_set>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#endif

void console_init(console_state& con_st) {
#if defined(_WIN32)
  // Windows-specific console initialization
  DWORD dwMode = 0;
  con_st.hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if (con_st.hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(con_st.hConsole, &dwMode)) {
    con_st.hConsole = GetStdHandle(STD_ERROR_HANDLE);
    if (con_st.hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(con_st.hConsole, &dwMode))) {
      con_st.hConsole = NULL;
    }
  }
  if (con_st.hConsole) {
    // Enable ANSI colors on Windows 10+
    if (con_st.use_color && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
      SetConsoleMode(con_st.hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    // Set console output codepage to UTF8
    SetConsoleOutputCP(CP_UTF8);
  }
  HANDLE hConIn = GetStdHandle(STD_INPUT_HANDLE);
  if (hConIn != INVALID_HANDLE_VALUE && GetConsoleMode(hConIn, &dwMode)) {
    // Set console input codepage to UTF16
    _setmode(_fileno(stdin), _O_WTEXT);

    // Turn off ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
    dwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
    SetConsoleMode(hConIn, dwMode);
  }
#else
  // POSIX-specific console initialization
  struct termios new_termios;
  tcgetattr(STDIN_FILENO, &con_st.prev_state);
  new_termios = con_st.prev_state;
  new_termios.c_lflag &= ~(ICANON | ECHO);
  new_termios.c_cc[VMIN] = 1;
  new_termios.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

  con_st.tty = fopen("/dev/tty", "w+");
  if (con_st.tty != nullptr) {
    con_st.out = con_st.tty;
  }

  setlocale(LC_ALL, "");
#endif
}

void console_cleanup(console_state& con_st) {
  // Reset console color
  console_set_color(con_st, CONSOLE_COLOR_DEFAULT);

#if !defined(_WIN32)
  if (con_st.tty != nullptr) {
    con_st.out = stdout;
    fclose(con_st.tty);
    con_st.tty = nullptr;
  }
  // Restore the terminal settings on POSIX systems
  tcsetattr(STDIN_FILENO, TCSANOW, &con_st.prev_state);
#endif
}

/* Keep track of current color of output, and emit ANSI code if it changes. */
void console_set_color(console_state& con_st, console_color_t color) {
  if (con_st.use_color && con_st.color != color) {
    fflush(stdout);
    switch (color) {
      case CONSOLE_COLOR_DEFAULT:
        fprintf(con_st.out, ANSI_COLOR_RESET);
        break;
      case CONSOLE_COLOR_PROMPT:
        fprintf(con_st.out, ANSI_COLOR_YELLOW);
        break;
      case CONSOLE_COLOR_USER_INPUT:
        fprintf(con_st.out, ANSI_BOLD ANSI_COLOR_GREEN);
        break;
    }
    con_st.color = color;
    fflush(con_st.out);
  }
}

char32_t getchar32() {
  wchar_t wc = getwchar();
  if (static_cast<wint_t>(wc) == WEOF) {
    return WEOF;
  }

#if WCHAR_MAX == 0xFFFF
  if ((wc >= 0xD800) && (wc <= 0xDBFF)) {  // Check if wc is a high surrogate
    wchar_t low_surrogate = getwchar();
    if ((low_surrogate >= 0xDC00) && (low_surrogate <= 0xDFFF)) {  // Check if the next wchar is a low surrogate
      return (static_cast<char32_t>(wc & 0x03FF) << 10) + (low_surrogate & 0x03FF) + 0x10000;
    }
  }
  if ((wc >= 0xD800) && (wc <= 0xDFFF)) {  // Invalid surrogate pair
    return 0xFFFD;                         // Return the replacement character U+FFFD
  }
#endif

  return static_cast<char32_t>(wc);
}

void pop_cursor(console_state& con_st) {
#if defined(_WIN32)
  if (con_st.hConsole != NULL) {
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo);

    COORD newCursorPosition = bufferInfo.dwCursorPosition;
    if (newCursorPosition.X == 0) {
      newCursorPosition.X = bufferInfo.dwSize.X - 1;
      newCursorPosition.Y -= 1;
    } else {
      newCursorPosition.X -= 1;
    }

    SetConsoleCursorPosition(con_st.hConsole, newCursorPosition);
    return;
  }
#endif
  putc('\b', con_st.out);
}

int estimateWidth(char32_t codepoint) {
#if defined(_WIN32)
  return 1;
#else
  return wcwidth(codepoint);
#endif
}

int put_codepoint(console_state& con_st, const char* utf8_codepoint, size_t length, int expectedWidth) {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
  if (!GetConsoleScreenBufferInfo(con_st.hConsole, &bufferInfo)) {
    // go with the default
    return expectedWidth;
  }
  COORD initialPosition = bufferInfo.dwCursorPosition;
  DWORD nNumberOfChars = length;
  WriteConsole(con_st.hConsole, utf8_codepoint, nNumberOfChars, &nNumberOfChars, NULL);

  CONSOLE_SCREEN_BUFFER_INFO newBufferInfo;
  GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);

  // Figure out our real position if we're in the last column
  if (utf8_codepoint[0] != 0x09 && initialPosition.X == newBufferInfo.dwSize.X - 1) {
    DWORD nNumberOfChars;
    WriteConsole(con_st.hConsole, &" \b", 2, &nNumberOfChars, NULL);
    GetConsoleScreenBufferInfo(con_st.hConsole, &newBufferInfo);
  }

  int width = newBufferInfo.dwCursorPosition.X - initialPosition.X;
  if (width < 0) {
    width += newBufferInfo.dwSize.X;
  }
  return width;
#else
  // we can trust expectedWidth if we've got one
  if (expectedWidth >= 0 || con_st.tty == nullptr) {
    fwrite(utf8_codepoint, length, 1, con_st.out);
    return expectedWidth;
  }

  fputs("\033[6n", con_st.tty);  // Query cursor position
  int x1, x2, y1, y2;
  int results = 0;
  results = fscanf(con_st.tty, "\033[%d;%dR", &y1, &x1);

  fwrite(utf8_codepoint, length, 1, con_st.tty);

  fputs("\033[6n", con_st.tty);  // Query cursor position
  results += fscanf(con_st.tty, "\033[%d;%dR", &y2, &x2);

  if (results != 4) {
    return expectedWidth;
  }

  int width = x2 - x1;
  if (width < 0) {
    // Calculate the width considering text wrapping
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width += w.ws_col;
  }
  return width;
#endif
}

void replace_last(console_state& con_st, char ch) {
#if defined(_WIN32)
  pop_cursor(con_st);
  put_codepoint(con_st, &ch, 1, 1);
#else
  fprintf(con_st.out, "\b%c", ch);
#endif
}

void append_utf8(char32_t ch, std::string& out) {
  if (ch <= 0x7F) {
    out.push_back(static_cast<unsigned char>(ch));
  } else if (ch <= 0x7FF) {
    out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
    out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
  } else if (ch <= 0xFFFF) {
    out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
    out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
    out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
  } else if (ch <= 0x10FFFF) {
    out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
    out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
    out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
    out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
  } else {
    // Invalid Unicode code point
  }
}

// Helper function to remove the last UTF-8 character from a string
void pop_back_utf8_char(std::string& line) {
  if (line.empty()) {
    return;
  }

  size_t pos = line.length() - 1;

  // Find the start of the last UTF-8 character (checking up to 4 bytes back)
  for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
    if ((line[pos] & 0xC0) != 0x80) break;  // Found the start of the character
  }
  line.erase(pos);
}

bool console_readline(console_state& con_st, std::string& line) {
  console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
  if (con_st.out != stdout) {
    fflush(stdout);
  }

  line.clear();
  std::vector<int> widths;
  bool is_special_char = false;
  bool end_of_stream = false;

  char32_t input_char;
  while (true) {
    fflush(con_st.out);  // Ensure all output is displayed before waiting for input
    input_char = getchar32();

    if (input_char == '\r' || input_char == '\n') {
      break;
    }

    if (input_char == WEOF || input_char == 0x04 /* Ctrl+D*/) {
      end_of_stream = true;
      break;
    }

    if (is_special_char) {
      console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
      replace_last(con_st, line.back());
      is_special_char = false;
    }

    if (input_char == '\033') {  // Escape sequence
      char32_t code = getchar32();
      if (code == '[' || code == 0x1B) {
        // Discard the rest of the escape sequence
        while ((code = getchar32()) != WEOF) {
          if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
            break;
          }
        }
      }
    } else if (input_char == 0x08 || input_char == 0x7F) {  // Backspace
      if (!widths.empty()) {
        int count;
        do {
          count = widths.back();
          widths.pop_back();
          // Move cursor back, print space, and move cursor back again
          for (int i = 0; i < count; i++) {
            replace_last(con_st, ' ');
            pop_cursor(con_st);
          }
          pop_back_utf8_char(line);
        } while (count == 0 && !widths.empty());
      }
    } else {
      int offset = line.length();
      append_utf8(input_char, line);
      int width = put_codepoint(con_st, line.c_str() + offset, line.length() - offset, estimateWidth(input_char));
      if (width < 0) {
        width = 0;
      }
      widths.push_back(width);
    }

    if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
      console_set_color(con_st, CONSOLE_COLOR_PROMPT);
      replace_last(con_st, line.back());
      is_special_char = true;
    }
  }

  bool has_more = con_st.multiline_input;
  if (is_special_char) {
    replace_last(con_st, ' ');
    pop_cursor(con_st);

    char last = line.back();
    line.pop_back();
    if (last == '\\') {
      line += '\n';
      fputc('\n', con_st.out);
      has_more = !has_more;
    } else {
      // model will just eat the single space, it won't act as a space
      if (line.length() == 1 && line.back() == ' ') {
        line.clear();
        pop_cursor(con_st);
      }
      has_more = false;
    }
  } else {
    if (end_of_stream) {
      has_more = false;
    } else {
      line += '\n';
      fputc('\n', con_st.out);
    }
  }

  fflush(con_st.out);
  return has_more;
}
