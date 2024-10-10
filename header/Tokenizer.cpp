#include "Tokenizer.h"

std::vector<std::string> customTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    bool inQuotes = false;
    std::ostringstream currentToken;

    char ch;
    while (iss.get(ch)) {
        if (ch == '"') {
            inQuotes = !inQuotes;
        } else if (ch == ',' && !inQuotes) {
            tokens.push_back(currentToken.str());
            currentToken.str("");
            currentToken.clear();
        } else {
            currentToken << ch;
        }
    }

    // Add the last token
    if (!currentToken.str().empty()) {
        tokens.push_back(currentToken.str());
    }

    // Trim whitespace and remove quotes from each token
    for (auto& token : tokens) {
        token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
        token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
            token = token.substr(1, token.size() - 2);
        }
    }

    return tokens;
}

// Usage in tokenizeData function
std::vector<std::vector<std::string>> tokenizeData(const std::vector<std::string>& textData) {
    std::vector<std::vector<std::string>> tokenizedData;
    for (const std::string& text : textData) {
        tokenizedData.push_back(customTokenize(text));
    }
    return tokenizedData;
}