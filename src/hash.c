#include "hash.h"

#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <openssl/evp.h>
#include <openssl/sha.h>

#include "error.h"

static void build_hash (void *data, size_t bytes, char *hash)
{
  int i;
  unsigned char sha[SHA256_DIGEST_LENGTH] = {0};
  char const alpha[]                      = "0123456789abcdef";

  EVP_Digest (data, bytes, sha, NULL, EVP_sha256 (), NULL);

  for (i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    hash[2 * i]     = alpha[sha[i] >> 4];
    hash[2 * i + 1] = alpha[sha[i] & 0xF];
  }
  hash[2 * i] = '\0';
}

void build_hash_and_store_to_file (void *buffer, size_t len,
                                   const char *filename)
{
  char hash[2 * SHA256_DIGEST_LENGTH + 1];

  int fd = open (filename, O_CREAT | O_WRONLY | O_TRUNC, 0666);
  if (fd == -1)
    exit_with_error ("Cannot create \"%s\" file (%s)", filename,
                     strerror (errno));

  build_hash (buffer, len, hash);

  if (write (fd, hash, 2 * SHA256_DIGEST_LENGTH) == -1)
    exit_with_error ("Cannot write to \"%s\" file (%s)", filename,
                     strerror (errno));
  close (fd);
  printf ("SHA256 key stored in %s\n", filename);
}
