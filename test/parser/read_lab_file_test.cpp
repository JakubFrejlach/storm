/*
 * read_lab_file_test.cpp
 *
 *  Created on: 12.09.2012
 *      Author: Thomas Heinemann
 */

#include "gtest/gtest.h"
#include "src/dtmc/labelling.h"
#include "src/parser/read_lab_file.h"
#include "src/exceptions/file_IO_exception.h"
#include "src/exceptions/wrong_file_format.h"

TEST(ReadLabFileTest, NonExistingFileTest) {
   //No matter what happens, please don't create a file with the name "nonExistingFile.not"! :-)
   ASSERT_THROW(mrmc::parser::read_lab_file(0,"nonExistingFile.not"), mrmc::exceptions::file_IO_exception);
}

TEST(ReadLabFileTest, ParseTest) {
   //This test is based on a testcase from the original MRMC.
   mrmc::dtmc::labelling* labelling;

   //Parsing the file
   ASSERT_NO_THROW(labelling = mrmc::parser::read_lab_file(12,"test/parser/lab_files/pctl_general_input_01.lab"));

   //Checking whether all propositions are in the labelling

   char phi[] = "phi", psi[] = "psi", smth[] = "smth";

   ASSERT_TRUE(labelling->containsProposition(phi));
   ASSERT_TRUE(labelling->containsProposition(psi));
   ASSERT_TRUE(labelling->containsProposition(smth));

   //Testing whether all and only the correct nodes are labeled with "phi"
   ASSERT_TRUE(labelling->nodeHasProposition(phi,1));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,2));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,3));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,5));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,7));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,9));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,10));
   ASSERT_TRUE(labelling->nodeHasProposition(phi,11));

   ASSERT_FALSE(labelling->nodeHasProposition(phi,4));
   ASSERT_FALSE(labelling->nodeHasProposition(phi,6));

   //Testing whether all and only the correct nodes are labeled with "psi"
   ASSERT_TRUE(labelling->nodeHasProposition(psi,6));
   ASSERT_TRUE(labelling->nodeHasProposition(psi,7));
   ASSERT_TRUE(labelling->nodeHasProposition(psi,8));

   ASSERT_FALSE(labelling->nodeHasProposition(psi,1));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,2));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,3));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,4));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,5));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,9));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,10));
   ASSERT_FALSE(labelling->nodeHasProposition(psi,11));

   //Testing whether all and only the correct nodes are labeled with "smth"
   ASSERT_TRUE(labelling->nodeHasProposition(smth,4));
   ASSERT_TRUE(labelling->nodeHasProposition(smth,5));

   ASSERT_FALSE(labelling->nodeHasProposition(smth,1));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,2));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,3));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,6));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,7));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,8));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,9));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,10));
   ASSERT_FALSE(labelling->nodeHasProposition(smth,11));

   //Deleting the labelling
   delete labelling;
}
