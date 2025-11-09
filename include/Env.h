
/**
 * @file
 * @brief Get values from environment
 * @author Erwin Meza Vega <emezav@unicauca.edu.co>, <emezav@gmail.com>
 * @copyright MIT License
 *
 */

#ifndef ENV_H
#define ENV_H

#include <string>

/**
 * @brief Environment
 */
class Env
{

public:
  /**
   * @brief Gets the value of an environment variable
   * @param key Variable name
   * @return Environment variable value | empty string
   */
  static std::string get(std::string key)
  {
    char *val;

    // Get environment variable value
    val = std::getenv(key.c_str());

    std::string retval = "";
    if (val != NULL)
    {
      retval = val;
    }
    return retval;
  }
};

#endif // ENV_H
