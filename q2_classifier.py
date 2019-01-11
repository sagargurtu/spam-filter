"""
Spam Filter (Naive Bayes Approach)

Author: Sagar Gurtu

Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>

Put TREC Corpus data in /trec/

"""
import sys
import os
from math import log10

treccorpus = False


class Corpus:
    """
    This class stores the dictionary for spam and ham features as well as stats
    """

    def __init__(self):
        self.spam_dict = dict()  # Spam Dictionary
        self.ham_dict = dict()  # Ham Dictionary

        self.spam_words = 0.0  # Total spam words count
        self.ham_words = 0.0  # Total ham words count
        self.unique_spam_words = 0.0  # Unique spam words count
        self.unique_ham_words = 0.0  # Unique ham words count

        self.spam_emails = 0.0  # Total spam emails
        self.ham_emails = 0.0  # Total ham emails

        self.spam_domains = 0.0  # Total spam domains count
        self.ham_domains = 0.0  # Total ham domains count
        self.unique_spam_domains = 0.0  # Unique spam domains count
        self.unique_ham_domains = 0.0  # Unique ham domains count

        '''
        self.spam_phrases = 0.0
        self.ham_phrases = 0.0
        self.unique_spam_phrases = 0.0
        self.unique_ham_phrases = 0.0
        '''

    def _add_spam(self, feature, count):
        """
        Adds spam feature to the dictionary
        :param feature:
        :param count:
        :return:
        """
        self.spam_dict[feature] = self.spam_dict.get(feature, 0.0) + float(count)

    def _add_ham(self, feature, count):
        """
        Adds ham feature to the dictionary
        :param feature:
        :param count:
        :return:
        """
        self.ham_dict[feature] = self.ham_dict.get(feature, 0.0) + float(count)

    def add_spam_word(self, word, count):
        """
        Adds spam word to the dictionary
        - Increments unique spam words count if haven't seen before
        - Increments total spam words count
        :param word:
        :param count:
        :return:
        """
        if word not in self.spam_dict:
            self.unique_spam_words += 1.0
        self._add_spam(word, count)
        self.spam_words += 1.0

    def add_ham_word(self, word, count):
        """
        Adds ham word to the dictionary
        - Increments unique ham words count if haven't seen before
        - Increments total ham words count
        :param word:
        :param count:
        :return:
        """
        if word not in self.ham_dict:
            self.unique_ham_words += 1.0
        self._add_ham(word, count)
        self.ham_words += 1.0

    def add_spam_domain(self, domain):
        """
        Adds spam domain to the dictionary
        - Increments unique spam domains count if haven't seen before
        - Increments total spam domains count
        :param domain:
        :return:
        """
        if domain not in self.spam_dict:
            self.unique_spam_domains += 1.0
        self._add_spam(domain, 1.0)
        self.spam_domains += 1.0

    def add_ham_domain(self, domain):
        """
        Adds ham domain to the dictionary
        - Increments unique ham domains count if haven't seen before
        - Increments total ham domains count
        :param domain:
        :return:
        """
        if domain not in self.ham_dict:
            self.unique_ham_domains += 1.0
        self._add_ham(domain, 1.0)
        self.ham_domains += 1.0

    # Commented out code for phrases. Very little or no change in accuracy.
    '''
    def add_spam_phrases(self, phrases):
        for phrase in phrases:
            if phrase not in self.spam_dict:
                self.unique_spam_phrases += 1.0
            self._add_spam(phrase, 1.0)
            self.spam_phrases += 1.0

    def add_ham_phrases(self, phrases):
        for phrase in phrases:
            if phrase not in self.ham_dict:
                self.unique_ham_phrases += 1.0
            self._add_ham(phrase, 1.0)
            self.ham_phrases += 1.0
    '''

    def add_spam_email(self):
        """
        Increments total spam emails count
        :return:
        """
        self.spam_emails += 1.0

    def add_ham_email(self):
        """
        Increments total ham emails count
        :return:
        """
        self.ham_emails += 1.0

    def get_spam_count(self, word):
        """
        Gets count for spam feature
        :param word:
        :return:
        """
        return self.spam_dict.get(word, 0.0)

    def get_ham_count(self, word):
        """
        Gets count for ham feature
        :param word:
        :return:
        """
        return self.ham_dict.get(word, 0.0)

    def display_stats(self):
        """
        Prints Stats
        :return:
        """
        print "\n\t\t    TRAIN CORPUS\n\t\t    ------------\n"
        print "\t\tSpam Emails: " + str(int(self.spam_emails))
        print "\t\t Ham Emails: " + str(int(self.ham_emails)) + "\n"
        print "\t\tTotal Words: " + str(int(self.spam_words + self.ham_words))
        print "\t\t Spam Words: " + str(int(self.spam_words))
        print "\t\t  Ham Words: " + str(int(self.ham_words)) + "\n"
        print "\t       Spam Domains: " + str(int(self.spam_domains))
        print "\t\tHam Domains: " + str(int(self.ham_domains))


def trec_file_exists(email_id):
    """
    Checks if TREC file exists on disk
    :param email_id:
    :return:
    """
    if not os.path.exists("trec/" + email_id):
        print "WARN: TREC file " + email_id + " not found. Probable Reason: Removed due to infection."
        return False
    return True


def extract_domain(email_id):
    """
    Gets domain name from email_id
    :param email_id: Unique identifier for email
    :return:
    """

    domain = None
    # If email file cannot be found, return
    if not trec_file_exists(email_id):
        return domain

    # Open email file, check each line for Sender email and if found, extract domain name
    email = open("trec/" + email_id)
    for line in email:
        if line[0:4] == "From":
            if line.rfind("@") != -1:
                domain = line[line.rfind("@") + 1: line.rfind(">")].split(" ")[0].lower()
            email.close()
            break

    return domain


# Code for phrases. Very little or no change in accuracy.
def extract_phrases(email_id):
    """
    Get spam phrases from email
    :param email_id:
    :return:
    """
    phrases = []
    # If email body cannot be found, return
    if not trec_file_exists(email_id):
        return phrases

    spam_phrases = ["free!", "only $", "free money", "for just $", "!!!", "$$$", "save $",
                    "money back", "earn $", "make $", "work at home", "work from home",
                    "no fees", "for just", "click here", "100%", "50%", "100% satisfied",
                    "call now", "once in lifetime", "don't miss"]

    # Open email file, check each line for Sender email and if found, extract domain name
    email = open("trec/" + email_id).read().replace('\n', '').lower()
    for spam_phrase in spam_phrases:
        if spam_phrase in email:
            phrases.append(spam_phrase)

    return phrases


def generate_features(corpus, email_id, label):
    """
    If TREC Corpus exists, get domain name from email_id and add to corpus
    depending upon email being spam/ham
    :param corpus: train metadata
    :param email_id: unique identifier for email
    :param label: spam/ham
    :return:
    """
    if treccorpus:
        domain = extract_domain(email_id)
        if domain is not None:
            corpus.add_spam_domain(domain) if label == "spam" else corpus.add_ham_domain(domain)
        # Commented out code for phrases. Very little or no change in accuracy.
        '''
        phrases = extract_phrases(email_id)
        corpus.add_spam_phrases(phrases) if tokens[1] == "spam" else corpus.add_ham_phrases(phrases)
        '''


def train(train_file):
    """
    Builds the train corpus including the dictionaries for spam and ham words
    - If TREC Corpus exists, collects domain features

    :param train_file: path of train dataset
    :return:
    """

    # Create a train corpus to store data at one place
    corpus = Corpus()
    # Open the train dataset file
    training_data = open(train_file, "r")

    # For each email metadata in train dataset
    for email in training_data:
        tokens = email.split(" ")
        email_id = tokens[0]  # Unique identifier for email

        # If current email is spam, increment spam emails count else increment ham emails count
        corpus.add_spam_email() if tokens[1] == "spam" else corpus.add_ham_email()

        # For each word in the email, add to corpus depending upon email being spam/ham
        for word_index in range(2, len(tokens), 2):
            if tokens[1] == "spam":
                corpus.add_spam_word(tokens[word_index], tokens[word_index + 1])
            else:
                corpus.add_ham_word(tokens[word_index], tokens[word_index + 1])

        generate_features(corpus, email_id, tokens[1])

    training_data.close()
    return corpus


def compute_probabilities(corpus, tokens):
    """
    Compute spam and ham probabilities for the email
    :param corpus:
    :param tokens:
    :return:
    """

    a = 1e-10  # Alpha parameter for Laplace Smoothing
    email_id = tokens[0]

    # Calculate P(spam|email) = pi(P(email|spam)) * P(spam), ignoring denominator P(email)
    #        and P(ham|email) = pi(P(email|ham)) * P(ham), ignoring denominator P(email)
    #
    # Also, pi(P(email|spam)) = pi(P(word|spam)^count), conditional independence
    #
    # First compute P(ck) i.e.
    #       Prob(spam) = #(spam emails in train) / #(total emails in train)
    #       Prob(ham) = #(ham emails in train) / #(total emails in train)
    p_spam = log10(corpus.spam_emails / (corpus.spam_emails + corpus.ham_emails))
    p_ham = log10(corpus.ham_emails / (corpus.spam_emails + corpus.ham_emails))

    # Now compute pi{P(xi|ck)^count}
    # Prob(word|spam)^count = (#(word as spam in train) / #(spam words in train)) ^ #(word in test email)
    # Prob(word|ham)^count = (#(word as ham in train) / #(ham words in train)) ^ #(word in test email)
    #
    # After adding laplace smoothing
    # Prob(word|spam)^count = ((#(word as spam in train) + a) /
    #                          (#(spam words in train) + a * #(unique spam words in train))) ^
    #                         #(word in test email)
    # Similar for ham
    #
    # Taking logs converts products into sums
    # log(pi{P(xi|ck)^count}) = sum(log(P(xi|ck)) * count)
    for word_index in range(2, len(tokens), 2):
        p_spam += log10((corpus.get_spam_count(tokens[word_index]) + a) / (
                corpus.spam_words + a * corpus.unique_spam_words)) * float(tokens[word_index + 1])
        p_ham += log10((corpus.get_ham_count(tokens[word_index]) + a) / (
                corpus.ham_words + a * corpus.unique_ham_words)) * float(tokens[word_index + 1])

    # If TREC Corpus is present, get domain name and compute P(domain|ck)
    #   Prob(domain|spam) = (#(times domain classified as spam) + a) /
    #                       (#(spam domains) + a * #(unique spam domains))
    #   Similar for ham
    if treccorpus:
        domain = extract_domain(email_id)
        if domain is not None:
            p_spam += log10((corpus.get_spam_count(domain) + a) / (
                    corpus.spam_domains + a * corpus.unique_spam_domains))
            p_ham += log10((corpus.get_ham_count(domain) + a) / (
                    corpus.ham_domains + a * corpus.unique_ham_domains))

        # Commented out code for phrases. Very little or no change in accuracy.
        '''
        phrases = extract_phrases(email_id)
        for phrase in phrases:
            p_spam += log10((corpus.get_spam_count(phrase) + a) / (
                             corpus.spam_phrases + a * corpus.unique_spam_phrases))
            p_ham += log10((corpus.get_ham_count(phrase) + a) / (
                            corpus.ham_phrases + a * corpus.unique_ham_phrases))
        '''

    return p_spam, p_ham


def test(corpus, test_file, output_file):
    """
    Classifies test data using Naive Bayes
    :param corpus: train corpus metadata
    :param test_file: path of test dataset
    :param output_file: path of output file
    :return:
    """
    # Open test dataset and output files
    testing_data = open(test_file, "r")
    output_pred = open(output_file, "w")

    # Spam is considered Positive and Ham is considered Negative below
    spam_pred = 0.0  # Spam Probability for an email
    ham_pred = 0.0  # Ham Probability for an email
    true_positive = 0.0  # Count of correctly identified Spam (True Positive)
    true_negative = 0.0  # Count of correctly identified Ham (True Negative)
    false_negative = 0.0  # Count of incorrectly identified Spam (False Negative)
    false_positive = 0.0  # Count of incorrectly identified Ham (False Positive)
    ambiguous = 0.0  # Ambiguous prediction count

    # For each email metadata in test set
    for email in testing_data:
        tokens = email.split(" ")
        email_id = tokens[0]
        label = tokens[1]  # Actual label for email
        result = "spam"

        # Compute spam and ham probabilities for the email
        p_spam, p_ham = compute_probabilities(corpus, tokens)

        # Now we have P(spam|email) and P(ham|email)
        # Check which is greater and assign result accordingly
        # If equal, set as ambiguous
        if p_spam > p_ham:
            spam_pred += 1.0
            if label == "spam":
                true_positive += 1.0
            else:
                false_positive += 1.0
        elif p_ham > p_spam:
            result = "ham"
            ham_pred += 1.0
            if label == "ham":
                true_negative += 1.0
            else:
                false_negative += 1.0
        else:
            result = "ham"  # it is better to assign legitimate in case of ambiguity
            ambiguous += 1.0
            if label == "spam":
                false_negative += 1.0
            else:
                true_negative += 1.0

        # Write to output file
        output_pred.write(email_id + "," + result + "\n")

    actual_spam = true_positive + false_negative  # True Positive + False Negative
    actual_ham = true_negative + false_positive  # True Negative + False Positive

    # Print results
    # Precision = True Positive / (True Positive + False Positive)
    # Recall = True Positive / (True Positive + False Negative)
    # Accuracy = (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)

    junk_precision = true_positive * 100 / spam_pred
    junk_recall = true_positive * 100 / actual_spam
    legitimate_precision = true_negative * 100 / ham_pred
    legitimate_recall = true_negative * 100 / actual_ham
    accuracy = (true_positive + true_negative) * 100 / (actual_spam + actual_ham)
    f_score = (2 * junk_precision * junk_recall) / (junk_precision + junk_recall)

    print "\n\t\t\t     TEST RESULTS\n\t\t\t     ------------\n"
    print "\tPred Spam: " + str(int(spam_pred)) + "\t  Actual Spam: " + str(int(actual_spam)) + "\tCorrect: " + str(
        int(true_positive)) + "\tIncorrect: " + str(int(false_negative))
    print "\t Pred Ham: " + str(int(ham_pred)) + "\t   Actual Ham: " + str(int(actual_ham)) + "\tCorrect: " + str(
        int(true_negative)) + "\tIncorrect: " + str(int(false_positive)) + "\n"
    print "\t\t      Junk Precision: " + str(junk_precision)
    print "\t\t         Junk Recall: " + str(junk_recall)
    print "\t\tLegitimate Precision: " + str(legitimate_precision)
    print "\t\t   Legitimate Recall: " + str(legitimate_recall)
    print "\t\t            Accuracy: " + str(accuracy)
    print "\t\t        Junk F-Score: " + str(f_score) + "\n"

    testing_data.close()
    output_pred.close()


def check_arguments(arguments):
    """
    Validates the arguments passed to q2_classifier.py
    - Checks usage syntax
    - Checks if train and test datasets exist
    - Checks if TREC Corpus exists

    :param arguments:
    :return:
    """
    global treccorpus

    if len(arguments) != 7 or arguments[1] != "-f1" or arguments[3] != "-f2" or arguments[5] != "-o":
        print "USAGE: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>"
        exit(1)

    if not os.path.isfile(arguments[2]):
        print "ERROR: Train Dataset not found."
        exit(1)

    if not os.path.isfile(arguments[4]):
        print "ERROR: Test Dataset not found."
        exit(1)

    if os.path.exists("trec/"):
        treccorpus = True
        print "\nINFO: TREC Public Spam Corpus found."
    else:
        print "\nINFO: TREC Public Spam Corpus not found."


if __name__ == "__main__":
    args = sys.argv
    check_arguments(args)

    print "\nINFO: Training..."
    corpus = train(args[2])
    corpus.display_stats()

    print "\n\nINFO: Classifying test data..."
    test(corpus, args[4], args[6])
