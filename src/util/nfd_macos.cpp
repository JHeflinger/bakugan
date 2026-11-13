// nativefiledialog on macOS. The nfd module ships libraries for Linux and Windows only, so here the calls the app
// makes (open, save, pick folder, last error) are answered by the system's own dialogs through osascript, the
// same way tinyfiledialogs does it on the Mac. Paths are returned malloc'd, as nfd does: the caller frees them.
#ifdef __APPLE__
#include <nfd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
    std::string s_Error;

    // an AppleScript string literal
    std::string Quote(const char* s) {
        std::string out = "\"";
        for (const char* c = s ? s : ""; *c; c++) {
            if (*c == '"' || *c == '\\') out += '\\';
            out += *c;
        }
        return out + "\"";
    }

    // a shell single-quoted argument
    std::string ShellQuote(const std::string& s) {
        std::string out = "'";
        for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
        return out + "'";
    }

    // nfd filters: "obj,gltf;txt" (commas within a group, semicolons between groups) -> every extension
    std::vector<std::string> Extensions(const char* filters) {
        std::vector<std::string> out;
        std::string cur;
        for (const char* c = filters ? filters : ""; ; c++) {
            if (*c == ',' || *c == ';' || *c == '\0') {
                if (!cur.empty()) out.push_back(cur);
                cur.clear();
                if (*c == '\0') break;
            } else if (*c != ' ' && *c != '.') {
                cur += *c;
            }
        }
        return out;
    }

    // runs an AppleScript that returns a POSIX path: OKAY with the path, CANCEL, or ERROR with its message
    nfdresult_t Run(const std::string& script, nfdchar_t** outPath) {
        *outPath = nullptr;
        std::string cmd = "/usr/bin/osascript -e " + ShellQuote(script) + " 2>&1";
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) { s_Error = "could not run osascript"; return NFD_ERROR; }
        std::string text;
        char buf[512];
        while (size_t n = fread(buf, 1, sizeof(buf), p)) text.append(buf, n);
        int status = pclose(p);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        if (status != 0) {
            if (text.find("-128") != std::string::npos) return NFD_CANCEL;   // "User canceled. (-128)"
            s_Error = text.empty() ? "osascript failed" : text;
            return NFD_ERROR;
        }
        if (text.empty()) return NFD_CANCEL;
        *outPath = strdup(text.c_str());
        return NFD_OKAY;
    }

    std::string Location(const nfdchar_t* defaultPath) {
        return defaultPath && *defaultPath ? " default location (POSIX file " + Quote(defaultPath) + ")" : "";
    }
}

extern "C" {
    nfdresult_t NFD_OpenDialog(const nfdchar_t* filterList, const nfdchar_t* defaultPath, nfdchar_t** outPath) {
        std::string types;
        for (const std::string& e : Extensions(filterList)) types += (types.empty() ? "" : ", ") + Quote(e.c_str());
        std::string script = "activate\nPOSIX path of (choose file" + (types.empty() ? "" : " of type {" + types + "}") +
                             Location(defaultPath) + ")";
        return Run(script, outPath);
    }

    nfdresult_t NFD_SaveDialog(const nfdchar_t* filterList, const nfdchar_t* defaultPath, nfdchar_t** outPath) {
        std::string script = "activate\nPOSIX path of (choose file name" + Location(defaultPath) + ")";
        nfdresult_t r = Run(script, outPath);
        // the Mac's save sheet doesn't add the extension: give it the first one the filter allows
        std::vector<std::string> ext = Extensions(filterList);
        if (r == NFD_OKAY && !ext.empty()) {
            const char* name = strrchr(*outPath, '/');
            name = name ? name + 1 : *outPath;
            if (!strchr(name, '.')) {
                std::string withExt = std::string(*outPath) + "." + ext[0];
                free(*outPath);
                *outPath = strdup(withExt.c_str());
            }
        }
        return r;
    }

    nfdresult_t NFD_PickFolder(const nfdchar_t* defaultPath, nfdchar_t** outPath) {
        return Run("activate\nPOSIX path of (choose folder" + Location(defaultPath) + ")", outPath);
    }

    const char* NFD_GetError(void) { return s_Error.empty() ? nullptr : s_Error.c_str(); }
}
#endif
