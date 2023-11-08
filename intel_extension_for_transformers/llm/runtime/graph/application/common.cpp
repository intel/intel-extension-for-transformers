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

#ifndef _WIN32
#include <ext/alloc_traits.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

// third-party utilities
// use your favorite implementations
#define DR_WAV_IMPLEMENTATION
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <codecvt>
#include <sstream>
#include <cassert>
#include <iostream>
#include <string>

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool isValidFilename(const std::string& filename) {
  std::ifstream infile(filename.c_str());
  return infile.good();
}

std::string gpt_random_prompt(std::mt19937& rng) {
  const int r = rng() % 10;
  switch (r) {
    case 0:
      return "So";
    case 1:
      return "Once upon a time";
    case 2:
      return "When";
    case 3:
      return "The";
    case 4:
      return "After";
    case 5:
      return "If";
    case 6:
      return "import";
    case 7:
      return "He";
    case 8:
      return "She";
    case 9:
      return "They";
    default:
      return "To";
  }

  return "The";
}

void quant_print_usage(int argc, char** argv, const quant_params& params) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  --model_file          path to the fp32 model\n");
  fprintf(stderr, "  --out_file            path to the quantized model\n");
  fprintf(stderr,
          "  --config              path to the configuration file (default: "
          ")\n");
  fprintf(stderr, "  --nthread             number of threads to use (default: 1)\n");
  fprintf(stderr, "  --weight_dtype        number of bits to use for quantization (default: int4)\n");
  fprintf(stderr, "  --alg                 quantization algorithm to use: sym/asym (default: sym)\n");
  fprintf(stderr, "  --group_size          group size (default: 32)\n");
  fprintf(stderr, "  --scale_dtype         fp32/bf16 type for scales (default: fp32)\n");
  fprintf(stderr, "  --compute_dtype       data type of Gemm computation: int8/bf16/fp32 (default: int8)\n");
  fprintf(stderr, "  --use_ggml            enable ggml for quantization and inference\n");
  fprintf(stderr,
          "  --model_name          model name like falcon / llama (default: "
          "unknown)\n");
  fprintf(stderr, "\n");
}

bool quant_params_parse(int argc, char** argv, quant_params& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--model_file") {
      params.model_file = argv[++i];
    } else if (arg == "--out_file") {
      params.out_file = argv[++i];
    } else if (arg == "--config") {
      params.config = argv[++i];
    } else if (arg == "--nthread") {
      params.nthread = std::stoi(argv[++i]);
    } else if (arg == "--weight_dtype") {
      params.weight_dtype = argv[++i];
    } else if (arg == "--alg") {
      params.alg = argv[++i];
    } else if (arg == "--group_size") {
      params.group_size = std::stoi(argv[++i]);
    } else if (arg == "--scale_dtype") {
      params.scale_dtype = argv[++i];
    } else if (arg == "--compute_dtype") {
      params.compute_dtype = argv[++i];
    } else if (arg == "--use_ggml") {
      params.use_ggml = true;
    } else if (arg == "--model_name") {
      params.model_name = argv[++i];
      model_archs mt = model_name_to_arch::init().find(params.model_name);
      if (mt == MODEL_UNKNOWN) {
        exit(0);
      } else {
        params.model_arch = mt;
      }
    } else if (arg == "-h" || arg == "--help") {
      quant_print_usage(argc, argv, params);
      exit(0);
    } else {
      quant_print_usage(argc, argv, params);
      fprintf(stderr, "unrecognized arguments: %s", arg.c_str());
      exit(0);
    }
  }

  return true;
}

ne_ftype quant_params_to_ftype(const quant_params& params) {
  if (params.use_ggml) {
    if (params.weight_dtype == "int4") {
      if (params.alg == "sym") {
        return NE_FTYPE_MOSTLY_Q4_0;
      } else {
        return NE_FTYPE_MOSTLY_Q4_1;
      }
    } else if (params.weight_dtype == "int5") {
      if (params.alg == "sym") {
        return NE_FTYPE_MOSTLY_Q5_0;
      } else {
        return NE_FTYPE_MOSTLY_Q5_1;
      }
    } else if (params.weight_dtype == "int8") {
      return NE_FTYPE_MOSTLY_Q8_0;
    }
  } else {
    return NE_FTYPE_MOSTLY_Q_JBLAS;
  }
  return NE_FTYPE_UNKNOWN;
}

ne_type quant_params_to_type(const quant_params& params) {
  if (params.use_ggml) {
    if (params.weight_dtype == "int4") {
      if (params.alg == "sym") {
        return NE_TYPE_Q4_0;
      } else {
        return NE_TYPE_Q4_1;
      }
    } else if (params.weight_dtype == "int5") {
      if (params.alg == "sym") {
        return NE_TYPE_Q5_0;
      } else {
        return NE_TYPE_Q5_1;
      }
    } else if (params.weight_dtype == "int8") {
      return NE_TYPE_Q8_0;
    }
  } else {
    return NE_TYPE_JBLAS;
  }
  return NE_TYPE_F32;
}

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

std::string build_prompt_glm2(const std::vector<std::string>& history) {
  std::ostringstream oss_prompt;
  for (size_t i = 0; i < history.size(); i += 2) {
    oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << history[i] << "\n\n答：";
    if (i < history.size() - 1) {
      oss_prompt << history[i + 1] << "\n\n";
    }
  }
  return oss_prompt.str();
}

std::string build_prompt_glm1(const std::vector<std::string>& history) {
  std::ostringstream oss_prompt;
  if (history.size() == 1) {
    oss_prompt << history.front();
  } else {
    for (size_t i = 0; i < history.size(); i += 2) {
      oss_prompt << "[Round " << i / 2 << "]\n问：" << history[i] << "\n答：";
      if (i < history.size() - 1) {
        oss_prompt << history[i + 1] << "\n";
      }
    }
  }
  return oss_prompt.str();
}

static std::string regex_replace(const std::string& input, const std::regex& regex,
                                 std::function<std::string(const std::smatch&)> format) {
  std::ostringstream oss;
  int last_index = 0;
  for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++) {
    oss << it->prefix() << format(*it);
    last_index = it->position() + it->length();
  }
  oss << input.substr(last_index);
  return oss.str();
}

std::string postprocess(const std::string& text) {
  std::string output;

  // newline token
  {
    static const std::regex pattern(R"(<n>)");
    output = std::regex_replace(text, pattern, "\n");
  }
  // tab token
  {
    static const std::regex pattern(R"(<\|tab\|>)");
    output = std::regex_replace(output, pattern, "\t");
  }
  // blank tokens
  {
    static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
    output =
        regex_replace(output, pattern, [](const std::smatch& sm) { return std::string(std::stoi(sm[1].str()), ' '); });
  }

  return output;
}
